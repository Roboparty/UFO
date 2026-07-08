"""Backward-compatible wrapper for the old MJLab reward inference module name.

Use `humanoidverse.reward_inference` for new commands.
"""

from humanoidverse.reward_inference import *  # noqa: F401,F403
from humanoidverse.reward_inference import main


if __name__ == "__main__":
    main()
