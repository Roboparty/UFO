"""Deployment configs, model contracts and task contexts."""

import json
import hashlib
from pathlib import Path

def load_yaml(path):
    import yaml
    path = Path(path).expanduser()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"{path} did not load as a mapping")
    return config


def load_robot_config(path):
    """Load a robot config with an optional single base file and joint overrides."""
    path = Path(path).expanduser()
    config = load_yaml(path)
    base = config.pop("base", None)
    if base is None:
        return config
    defaults = load_yaml(path.parent / base)
    if "base" in defaults:
        raise ValueError("Robot config supports one base file; nested/cyclic bases are not supported")
    for key, value in config.items():
        if isinstance(defaults.get(key), dict) and isinstance(value, dict):
            defaults[key] = {**defaults[key], **value}
        else:
            defaults[key] = value
    return defaults


def validate_task_config(config, robot_type):
    """Keep real teleop on the onboard host, including direct runtime callers."""
    if config.get("type") not in {"tracking", "reward", "goal"}:
        raise ValueError(f"Unsupported task type: {config.get('type')}")
    if config["type"] != "tracking":
        return
    source = config.get("ctx_source", "pkl")
    if source not in {"pkl", "zmq"}:
        raise ValueError(f"Invalid ctx_source: {source}")
    if source == "zmq" and robot_type == "g1_real":
        from urllib.parse import urlsplit

        address = urlsplit(str(config.get("ctx_zmq_addr", "tcp://127.0.0.1:28711")))
        if address.scheme != "tcp" or address.hostname not in {"127.0.0.1", "localhost", "::1"} or not address.port:
            raise ValueError("Teleop sim2real is onboard-only; ctx_zmq_addr must use localhost")


def resolve_context_path(model_path, exp_config, default_ctx_dir, fallback_ctx_dirs=()):
    ctx_path = Path(str(exp_config["ctx_path"])).expanduser()
    if ctx_path.is_absolute():
        return ctx_path
    model_export_dir = Path(model_path).expanduser().resolve().parent
    model_root = model_export_dir.parent
    dirs = [exp_config["ctx_dir"]] if "ctx_dir" in exp_config else [default_ctx_dir, *fallback_ctx_dirs]
    candidates = [model_root / directory / ctx_path for directory in dirs]
    candidates += [model_root / ctx_path, model_export_dir / ctx_path]
    candidates = list(dict.fromkeys(path.resolve() for path in candidates))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not resolve ctx_path={ctx_path}\nTried:\n{tried}")


def load_context(path):
    import joblib
    try:
        return joblib.load(path)
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ValueError("Context contains PyTorch tensors; re-export z as NumPy arrays with the updated UFO exporter") from exc
        raise


def validate_context(context):
    """Validate every stored vector and return the common latent dimension."""
    import numpy as np
    values = context.values() if isinstance(context, dict) else [context]
    arrays = [np.asarray(value) for value in values]
    if not arrays or any(value.ndim < 2 or value.size == 0 or not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("context must contain nonempty finite z arrays")
    dimensions = {value.shape[-1] for value in arrays}
    if len(dimensions) != 1:
        raise ValueError("context embeddings have inconsistent z dimensions")
    return dimensions.pop()


def select_contexts(contexts, exp_config):
    """Return reward/goal vectors and labels in the user's configured order."""
    import numpy as np
    if not isinstance(contexts, dict) or not contexts:
        raise ValueError("Reward/goal context must be a nonempty mapping")
    selected, names = [], []
    if exp_config["type"] == "reward":
        choices = exp_config.get("selected_rewards_filter_z")
        if isinstance(choices, dict):
            choices = [{"reward": name, "z_ids": ids} for name, ids in choices.items()]
        if not isinstance(choices, list):
            raise ValueError("selected_rewards_filter_z must be a list or mapping")
        for choice in choices:
            name = choice["reward"]
            if name not in contexts:
                raise ValueError(f"Unknown reward: {name}")
            for index in choice["z_ids"]:
                if type(index) is not int or not 0 <= index < len(contexts[name]):
                    raise ValueError(f"Invalid z index for reward {name!r}: {index}")
                selected.append(contexts[name][index])
                names.append(f'Reward="{name}"__Z_id={index}')
    elif exp_config["type"] == "goal":
        for name in dict.fromkeys(exp_config.get("selected_goals", contexts)):
            if name not in contexts:
                raise ValueError(f"Unknown goal: {name}")
            selected.append(contexts[name])
            names.append(name)
    else:
        raise ValueError(f"Unsupported context selection: {exp_config['type']}")
    if not selected:
        raise ValueError("No reward/goal contexts selected")
    vectors = [np.atleast_2d(value) for value in selected]
    if any(value.ndim != 2 or value.shape[0] != 1 or value.size == 0
           or not np.all(np.isfinite(value)) for value in vectors):
        raise ValueError("Each selected context must contain one finite z vector")
    return np.concatenate(vectors, axis=0), names


def load_policy_metadata(model_path):
    path = Path(model_path).with_suffix(".meta.json")
    if not path.is_file():
        raise ValueError(f"Missing training metadata: {path}; re-export the model and use its .deploy.yaml")
    metadata = json.loads(path.read_text())
    if metadata.get("onnx_sha256") != hashlib.sha256(Path(model_path).read_bytes()).hexdigest():
        raise ValueError("ONNX does not match its training metadata; re-export/copy the model and sidecars together")
    return metadata


def validate_policy_config(model_path, policy_config):
    """Check the training contract before opening any robot interface."""
    metadata = load_policy_metadata(model_path)
    from utils.strings import unitree_joint_names
    names = policy_config.get("policy_joint_names", [])
    if len(names) != 29 or set(names) != set(unitree_joint_names):
        raise ValueError("This deployment runtime supports Unitree G1 29-DoF only")
    expected = metadata.get("deployment")
    if not expected:
        raise ValueError("Model metadata has no deployment contract; re-export with the updated UFO exporter")
    if metadata.get("history_includes_current") is not False or metadata.get("history_oldest_first") is not False:
        raise ValueError("This runtime expects previous-frame history in newest-first order")
    if list(policy_config.get("observation", {}).get("policy", {})) != metadata["observation_order"]:
        raise ValueError("Observation order differs from training; use the model's .deploy.yaml")
    for key, value in expected.items():
        if policy_config.get(key) != value:
            raise ValueError(f"Deployment config differs from training: {key}; use the model's .deploy.yaml")
    return metadata
