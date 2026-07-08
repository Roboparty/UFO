"""Backward-compatible wrapper for the old MJLab training module name.

Use `humanoidverse.train` for new commands.
"""

from humanoidverse.train import *  # noqa: F401,F403
from humanoidverse.train import main


if __name__ == "__main__":
    main()
