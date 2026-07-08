"""Backward-compatible wrapper for the old MJLab goal inference module name.

Use `humanoidverse.goal_inference` for new commands.
"""

from humanoidverse.goal_inference import *  # noqa: F401,F403
from humanoidverse.goal_inference import main


if __name__ == "__main__":
    main()
