import json
from pathlib import Path
from typing import Any

from .fb.agent import FBAgent, FBModel
from .fb_cpr.agent import FBcprAgent, FBcprModel
from .fb_cpr_aux.agent import FBcprAuxAgent
from .fb_cpr_aux.model import FBcprAuxModel
from .gcr_rl_dist_aux.agent import GcrRlDistAuxAgent
from .gcr_rl_dist_aux.model import GcrRlDistAuxModel
from .tldr_dist_aux.agent import TldrDistAuxAgent

AGENT_NAME_TO_CLASS = {
    "FBAgent": FBAgent,
    "FBcprAgent": FBcprAgent,
    "FBcprAuxAgent": FBcprAuxAgent,
    "GcrRlDistAuxAgent": GcrRlDistAuxAgent,
    "TldrDistAuxAgent": TldrDistAuxAgent,
}
MODEL_NAME_TO_CLASS = {
    "FBModel": FBModel,
    "FBcprModel": FBcprModel,
    "FBcprAuxModel": FBcprAuxModel,
    "GcrRlDistAuxModel": GcrRlDistAuxModel,
}



def load_agent_from_checkpoint_dir(checkpoint_dir: str, device: str = "cpu") -> Any:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist.")
    with (checkpoint_dir / "config.json").open("r") as f:
        config = json.load(f)

    agent_name = config["name"]
    if agent_name not in AGENT_NAME_TO_CLASS:
        raise ValueError(f"Unknown agent name: {agent_name}. Available: {list(AGENT_NAME_TO_CLASS.keys())}")
    agent = AGENT_NAME_TO_CLASS[agent_name].load(checkpoint_dir, device=device)
    return agent


def load_model_from_checkpoint_dir(checkpoint_dir: str, device: str = "cpu") -> Any:
    checkpoint_dir = Path(checkpoint_dir)
    with (checkpoint_dir / "model" / "config.json").open("r") as f:
        config = json.load(f)
        
    model_name = config["name"]
    if model_name not in MODEL_NAME_TO_CLASS:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_NAME_TO_CLASS.keys())}")
    model = MODEL_NAME_TO_CLASS[model_name].load(checkpoint_dir / "model", device=device)
    return model
