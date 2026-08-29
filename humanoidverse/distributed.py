from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

_COLLECTIVE_BUCKET_BYTES = max(1, int(os.environ.get("UFO_COLLECTIVE_BUCKET_MB", "128"))) * 1024 * 1024


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def rank() -> int:
    return dist.get_rank() if is_distributed() else int(os.environ.get("RANK", "0"))


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else int(os.environ.get("WORLD_SIZE", "1"))


def wrap_distributed_stage(module: nn.Module, *, bucket_cap_mb: float = 25.0) -> nn.Module:
    """Wrap one optimizer stage while leaving the checkpointed model tree unchanged."""
    if not is_distributed():
        return module
    if bucket_cap_mb <= 0:
        raise ValueError(f"bucket_cap_mb must be positive, got {bucket_cap_mb}")

    parameters = tuple(module.parameters())
    if not parameters:
        raise ValueError("A distributed training stage must contain at least one parameter")
    device = parameters[0].device
    if any(parameter.device != device for parameter in parameters):
        raise ValueError("All parameters in a distributed training stage must be on one device")

    kwargs: dict[str, Any] = {
        "broadcast_buffers": False,
        "bucket_cap_mb": float(bucket_cap_mb),
        "find_unused_parameters": False,
        "gradient_as_bucket_view": True,
    }
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        kwargs.update(device_ids=[device_index], output_device=device_index)
    return DistributedDataParallel(module, **kwargs)


def barrier() -> None:
    if is_distributed():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def _all_reduce_buckets(tensors: Iterable[torch.Tensor], scale: float) -> None:
    """Average dense tensors using bounded flat collective buffers."""
    bucket: list[torch.Tensor] = []
    bucket_bytes = 0

    def flush() -> None:
        nonlocal bucket, bucket_bytes
        if not bucket:
            return
        if len(bucket) == 1 and bucket[0].is_contiguous():
            dist.all_reduce(bucket[0], op=dist.ReduceOp.SUM)
            bucket[0].mul_(scale)
        else:
            flat = torch.cat([tensor.reshape(-1) for tensor in bucket])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.mul_(scale)
            offset = 0
            for tensor in bucket:
                next_offset = offset + tensor.numel()
                tensor.copy_(flat[offset:next_offset].view_as(tensor))
                offset = next_offset
        bucket = []
        bucket_bytes = 0

    with torch.no_grad():
        for tensor in tensors:
            if tensor.numel() == 0:
                continue
            tensor_bytes = tensor.numel() * tensor.element_size()
            if bucket and bucket_bytes + tensor_bytes > _COLLECTIVE_BUCKET_BYTES:
                flush()
            if tensor_bytes > _COLLECTIVE_BUCKET_BYTES and tensor.is_contiguous():
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tensor.mul_(scale)
                continue
            bucket.append(tensor)
            bucket_bytes += tensor_bytes
        flush()


@torch.compiler.disable
def average_gradients(parameters: Iterable[torch.nn.Parameter]) -> None:
    if not is_distributed():
        return
    scale = 1.0 / float(dist.get_world_size())
    groups: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = defaultdict(list)
    for param in parameters:
        if param.grad is not None:
            groups[(param.grad.device, param.grad.dtype)].append(param.grad)
    for gradients in groups.values():
        _all_reduce_buckets(gradients, scale)


@torch.compiler.disable
def broadcast_module_state(module: torch.nn.Module, src: int = 0) -> None:
    if not is_distributed():
        return
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor.data, src=src)


@torch.compiler.disable
def sync_floating_buffers(module: torch.nn.Module, src: int = 0) -> None:
    if not is_distributed():
        return
    scale = 1.0 / float(dist.get_world_size())
    groups: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = defaultdict(list)
    for buffer in module.buffers():
        if buffer.dtype.is_floating_point or buffer.dtype.is_complex:
            groups[(buffer.device, buffer.dtype)].append(buffer.data)
        else:
            dist.broadcast(buffer.data, src=src)
    for buffers in groups.values():
        _all_reduce_buckets(buffers, scale)


@torch.compiler.disable
def broadcast_optimizer_state(optimizer: torch.optim.Optimizer, src: int = 0) -> None:
    if not is_distributed():
        return
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                dist.broadcast(value.data, src=src)


def agent_optimizers(agent: Any) -> list[torch.optim.Optimizer]:
    names = [
        "actor_optimizer",
        "backward_optimizer",
        "forward_optimizer",
        "critic_optimizer",
        "discriminator_optimizer",
        "aux_critic_optimizer",
    ]
    return [getattr(agent, name) for name in names if hasattr(agent, name)]


def broadcast_agent_state(agent: Any, src: int = 0) -> None:
    if not is_distributed():
        return
    broadcast_module_state(agent._model, src=src)
    for optimizer in agent_optimizers(agent):
        broadcast_optimizer_state(optimizer, src=src)


@torch.compiler.disable
def broadcast_object(value: Any, src: int = 0) -> Any:
    if not is_distributed():
        return value
    objects = [value if dist.get_rank() == src else None]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


@torch.compiler.disable
def all_gather_objects(value: Any) -> list[Any]:
    if not is_distributed():
        return [value]
    objects: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(objects, value)
    return objects


@torch.compiler.disable
def module_sync_report(module: torch.nn.Module, src: int = 0) -> dict[str, Any]:
    if not is_distributed():
        return {"world_size": 1, "rank": 0, "max_abs_diff_from_rank0": 0.0}

    tensors = list(module.parameters()) + [
        buffer for buffer in module.buffers() if buffer.dtype.is_floating_point or buffer.dtype.is_complex
    ]
    if tensors:
        device = tensors[0].device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_summary = torch.zeros(3, dtype=torch.float64, device=device)
    max_abs_diff = torch.zeros((), dtype=torch.float32, device=device)

    for tensor in tensors:
        detached = tensor.detach()
        local_summary[0] += detached.double().sum()
        local_summary[1] += detached.double().square().sum()
        local_summary[2] += detached.numel()
        reference = detached.clone()
        dist.broadcast(reference, src=src)
        if detached.numel() > 0:
            diff = (detached - reference).abs().max().float()
            max_abs_diff = torch.maximum(max_abs_diff, diff)

    dist.all_reduce(max_abs_diff, op=dist.ReduceOp.MAX)
    gathered = [torch.empty_like(local_summary) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_summary)
    summaries = torch.stack(gathered).cpu()
    return {
        "world_size": dist.get_world_size(),
        "rank": dist.get_rank(),
        "max_abs_diff_from_rank0": float(max_abs_diff.cpu().item()),
        "rank_param_buffer_sum": [float(x) for x in summaries[:, 0].tolist()],
        "rank_param_buffer_sqsum": [float(x) for x in summaries[:, 1].tolist()],
        "rank_param_buffer_numel": [int(x) for x in summaries[:, 2].tolist()],
    }


@torch.compiler.disable
def average_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    if not is_distributed():
        return dict(metrics)
    reduced: dict[str, Any] = {}
    scale = 1.0 / float(dist.get_world_size())
    for key, value in metrics.items():
        if torch.is_tensor(value):
            tensor = value.detach().clone()
            if not (tensor.dtype.is_floating_point or tensor.dtype.is_complex):
                tensor = tensor.float()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            reduced[key] = tensor * scale
        else:
            reduced[key] = value
    return reduced


@torch.compiler.disable
def reduce_metric_accumulators(
    totals: Mapping[str, torch.Tensor], counts: Mapping[str, int]
) -> dict[str, torch.Tensor]:
    """Reduce metric sums and counts once at log time instead of after every update."""
    if not totals:
        return {}
    keys = sorted(totals)
    first = totals[keys[0]]
    device = first.device if torch.is_tensor(first) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    packed = torch.empty((len(keys), 2), dtype=torch.float64, device=device)
    for index, key in enumerate(keys):
        value = totals[key].detach()
        packed[index, 0] = value.double().mean()
        packed[index, 1] = int(counts[key])
    if is_distributed():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return {
        key: (packed[index, 0] / packed[index, 1].clamp_min(1.0)).float()
        for index, key in enumerate(keys)
    }
