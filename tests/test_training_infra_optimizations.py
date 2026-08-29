from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from humanoidverse.agents.buffers.trajectory import TrajectoryDictBuffer
from humanoidverse.agents.envs.expert_motion_loader import (
    expert_buffer_cache_is_ready,
    expert_buffer_cache_spec,
    load_expert_buffer_cache,
    save_expert_buffer_cache,
)
from humanoidverse.agents.evaluations.humanoidverse_mjlab import (
    _joint_tracking_metrics,
    balanced_motion_chunks,
)
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgent
from humanoidverse.distributed import sync_floating_buffers, wrap_distributed_stage


class _FakeMotionLib:
    def __init__(self, motion_path, xml_path):
        self.m_cfg = OmegaConf.create(
            {
                "motion_file": str(motion_path),
                "asset": {"assetRoot": str(xml_path.parent), "assetFileName": xml_path.name},
                "body_names": ["pelvis", "torso"],
                "dof_names": ["joint_a", "joint_b"],
                "extend_config": [
                    {"joint_name": "head", "parent_name": "torso", "pos": [0, 0, 0.3], "rot": [1, 0, 0, 0]}
                ],
                "step_dt": 0.02,
            }
        )
        self._motion_data_keys = np.asarray(["motion-0"], dtype=object)
        self._num_unique_motions = 1
        self._num_motions = 1
        self.num_bodies = 2
        self.num_joints = 2
        self._device = "cpu"
        self.all_motions_loaded = True
        self._motion_lengths = torch.tensor([1.0])
        self._motion_fps = torch.tensor([50.0])
        self._motion_num_frames = torch.tensor([3])
        self.gts = torch.zeros(3, 2, 3)
        self.grs = torch.zeros(3, 2, 4)
        self.lrs = torch.zeros(3, 2, 4)
        self.length_starts = torch.tensor([0])
        self.refresh_count = 0

    def _refresh_sampling_batch_prob(self):
        self.refresh_count += 1


def _fake_env(tmp_path):
    motion_path = tmp_path / "motions.pkl"
    motion_path.write_bytes(b"motion-data")
    xml_path = tmp_path / "robot.xml"
    xml_path.write_text("<mujoco/>", encoding="utf-8")
    motion_lib = _FakeMotionLib(motion_path, xml_path)
    env = SimpleNamespace(
        _motion_lib=motion_lib,
        dt=0.02,
        num_dof=2,
        default_dof_pos=torch.zeros(1, 2),
        _max_local_self=torch.zeros(1, 7),
        config=SimpleNamespace(
            obs=OmegaConf.create(
                {
                    "root_height_obs": True,
                    "obs_auxiliary": {"history_actor": {"dof_pos": 4}},
                    "obs_dims": {"dof_pos": 2, "max_local_self": 7},
                }
            )
        ),
        _creation_config=SimpleNamespace(include_history_noaction=False),
    )
    return env, motion_path, xml_path


def test_expert_cache_fingerprint_covers_required_dependencies(tmp_path):
    env, motion_path, xml_path = _fake_env(tmp_path)
    agent_cfg = SimpleNamespace(model=SimpleNamespace(seq_length=5))

    _, original, _ = expert_buffer_cache_spec(env, agent_cfg, cache_root=tmp_path / "cache")
    xml_path.write_text("<mujoco model='changed'/>", encoding="utf-8")
    _, changed_xml, _ = expert_buffer_cache_spec(env, agent_cfg, cache_root=tmp_path / "cache")
    assert changed_xml != original

    motion_path.write_bytes(b"different-motion-data")
    _, changed_motion, _ = expert_buffer_cache_spec(env, agent_cfg, cache_root=tmp_path / "cache")
    assert changed_motion != changed_xml

    env._motion_lib.m_cfg.extend_config[0].pos = [0, 0, 0.4]
    _, changed_extend, _ = expert_buffer_cache_spec(env, agent_cfg, cache_root=tmp_path / "cache")
    assert changed_extend != changed_motion

    env.dt = 0.025
    _, changed_frequency, _ = expert_buffer_cache_spec(env, agent_cfg, cache_root=tmp_path / "cache")
    assert changed_frequency != changed_extend

    env.config.obs.obs_dims.max_local_self = 8
    _, changed_schema, _ = expert_buffer_cache_spec(env, agent_cfg, cache_root=tmp_path / "cache")
    assert changed_schema != changed_frequency


def test_expert_cache_round_trip_restores_buffer_and_full_fk(tmp_path):
    env, _, _ = _fake_env(tmp_path)
    agent_cfg = SimpleNamespace(model=SimpleNamespace(seq_length=2))
    episodes = [
        {
            "observation": {
                "state": torch.arange(18, dtype=torch.float32).reshape(3, 6),
                "last_action": torch.zeros(3, 2),
                "privileged_state": torch.ones(3, 7),
            },
            "terminated": torch.zeros(3, dtype=torch.bool),
            "truncated": torch.tensor([False, False, True]),
            "motion_id": torch.zeros(3, dtype=torch.long),
        }
    ]
    buffer = TrajectoryDictBuffer(episodes=episodes, seq_length=2, device="cpu")
    buffer.file_names = ["motion-0"]
    cache_dir, fingerprint, metadata = expert_buffer_cache_spec(env, agent_cfg, cache_root=tmp_path / "cache")
    save_expert_buffer_cache(env, buffer, cache_dir, fingerprint, metadata)
    assert expert_buffer_cache_is_ready(cache_dir, fingerprint)

    env._motion_lib.all_motions_loaded = False
    env._motion_lib.gts = None
    restored = load_expert_buffer_cache(env, cache_dir, fingerprint, device="cpu")
    assert restored.motion_ids == [0]
    assert restored.file_names == ["motion-0"]
    assert torch.equal(restored.storage["observation"]["state"], buffer.storage["observation"]["state"])
    assert env._motion_lib.all_motions_loaded
    assert isinstance(env._motion_lib.gts, torch.Tensor)
    assert env._motion_lib.refresh_count == 1


def test_balanced_motion_chunks_cover_every_motion_once():
    motion_ids = [0, 1, 2, 3, 4, 5, 6]
    lengths = {0: 100, 1: 90, 2: 80, 3: 70, 4: 60, 5: 50, 6: 40}
    rank_chunks = balanced_motion_chunks(motion_ids, lengths, chunk_size=2, world_size=3)
    assigned = [motion_id for chunks in rank_chunks for chunk in chunks for motion_id in chunk]
    assert sorted(assigned) == motion_ids
    assert len(assigned) == len(set(assigned))
    assert all(len(chunk) <= 2 for chunks in rank_chunks for chunk in chunks)


def test_parallel_joint_metric_matches_expected_distance():
    target = np.zeros((4, 2), dtype=np.float32)
    actual = np.asarray([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32)
    metrics = _joint_tracking_metrics(actual, target, invalid=False)
    assert np.isclose(metrics["distance"], 1.5)
    assert np.isclose(metrics["obs_state_distance"], metrics["distance"])
    assert np.isfinite(metrics["emd"])
    assert metrics["emd"] == metrics["obs_state_emd"]
    assert np.isscalar(metrics["mpjpe_l"])
    assert np.isscalar(metrics["vel_dist"])
    assert np.isscalar(metrics["accel_dist"])


def _ddp_gradient_worker(rank, init_method, output_queue):
    torch.distributed.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    try:
        module = torch.nn.Linear(2, 1, bias=False)
        module.register_buffer("running", torch.tensor([float(rank), float(rank + 2)]))
        module.register_buffer("count", torch.tensor(rank + 4, dtype=torch.int64))
        with torch.no_grad():
            module.weight.fill_(1.0)
        wrapped = wrap_distributed_stage(module, bucket_cap_mb=1.0)
        value = torch.tensor([[1.0 + 2.0 * rank, 2.0 + 2.0 * rank]])
        wrapped(value).sum().backward()
        sync_floating_buffers(module)
        output_queue.put(
            {
                "gradient": module.weight.grad.detach().tolist(),
                "running": module.running.tolist(),
                "count": module.count.item(),
            }
        )
    finally:
        torch.distributed.destroy_process_group()


def test_ddp_stage_averages_gradients_across_two_processes(tmp_path):
    context = torch.multiprocessing.get_context("fork")
    output_queue = context.SimpleQueue()
    init_method = f"file://{tmp_path / 'ddp-init'}"
    processes = [
        context.Process(target=_ddp_gradient_worker, args=(rank, init_method, output_queue))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    results = [output_queue.get() for _ in processes]
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    for result in results:
        assert np.allclose(result["gradient"], [[2.0, 3.0]])
        assert np.allclose(result["running"], [0.5, 2.5])
        assert result["count"] == 4


def test_fb_cpr_aux_registers_every_optimizer_stage_without_base_initializer():
    agent = FBcprAuxAgent.__new__(FBcprAuxAgent)
    agent._model = SimpleNamespace(
        _forward_map=torch.nn.Linear(2, 2),
        _backward_map=torch.nn.Linear(2, 2),
        _actor=torch.nn.Linear(2, 2),
        _critic=torch.nn.Linear(2, 2),
        _discriminator=torch.nn.Linear(2, 2),
        _aux_critic=torch.nn.Linear(2, 2),
    )
    agent.enable_distributed_gradient_sync(bucket_cap_mb=1.0)
    assert set(agent._distributed_training_stages) == {
        "fb",
        "actor",
        "critic",
        "discriminator",
        "aux_critic",
    }
