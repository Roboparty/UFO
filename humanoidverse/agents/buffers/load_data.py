from pathlib import Path
from typing import Callable
import json

import numpy as np
from torch.utils._pytree import tree_map

from humanoidverse.agents.buffers.trajectory import TrajectoryDictBuffer, TrajectoryDictBufferMultiDim
from humanoidverse.agents.buffers.transition import DictBuffer

def load_expert_trajectories(
    motions: str | Path,
    motions_root: str | Path,
    seq_length: int,
    device: str,
    obs_dict_mapper: Callable | None = None,
) -> TrajectoryDictBuffer:
    try:
        from humenv.misc.motionlib import canonicalize, load_episode_based_h5
    except ImportError as exc:
        raise ImportError(
            "load_expert_trajectories() requires optional dependency humenv. "
            "Install it with: uv sync --extra humenv"
        ) from exc

    with open(motions, "r") as txtf:
        h5files = [el.strip().replace(" ", "") for el in txtf.readlines()]
    episodes = []
    for h5 in h5files:
        h5 = canonicalize(h5, base_path=motions_root)
        _ep = load_episode_based_h5(h5, keys=None)
        for el in _ep:
            el["observation"] = tree_map(lambda x: x.astype(np.float32), el["observation"])
            if obs_dict_mapper is not None:
                assert isinstance(el["observation"], np.ndarray), (
                    "Received obs_dict_mapper but observation is not a numpy array. Is data stored already a dict? In that case you do not need the mapper"
                )
                el["observation"] = obs_dict_mapper(el["observation"])
            del el["file_name"]
        episodes.extend(_ep)
    buffer = TrajectoryDictBuffer(
        episodes,
        seq_length=seq_length,
        device=device,
    )

    return buffer

def load_buffer(path: str, device: str | None = None) -> DictBuffer:
    path = Path(path)
    with (path / "config.json").open() as f:
        loaded_config = json.load(f)
    target_class = loaded_config["__target__"]

    if target_class.endswith("DictBuffer"):
        return DictBuffer.load(path, device=device)
    elif target_class.endswith("TrajectoryDictBufferMultiDim"):
        return TrajectoryDictBufferMultiDim.load(path, device=device)