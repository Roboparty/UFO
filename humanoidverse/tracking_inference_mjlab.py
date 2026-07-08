"""Backward-compatible wrapper for the old MJLab tracking inference module name.

Use `humanoidverse.tracking_inference` for new commands.
"""

from humanoidverse.tracking_inference import *  # noqa: F401,F403
from humanoidverse.tracking_inference import main


if __name__ == "__main__":
    main()
