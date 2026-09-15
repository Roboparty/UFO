#!/usr/bin/env python3
"""Export the UFO FB backward latent encoder as a standalone ONNX model."""

from __future__ import annotations

from pathlib import Path

from humanoidverse.export.backward_encoder import export_backward_encoder_from_checkpoint


def main(
    model_folder: Path,
    output: Path | None = None,
    checkpoint_subdir: str = "checkpoint",
    device: str = "cuda",
    batch_size: int = 1,
    opset: int = 13,
    verify: bool = True,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    export_backward_encoder_from_checkpoint(
        model_folder=Path(model_folder),
        output=output,
        checkpoint_subdir=checkpoint_subdir,
        device=device,
        batch_size=batch_size,
        opset=opset,
        verify=verify,
        atol=atol,
        rtol=rtol,
    )


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
