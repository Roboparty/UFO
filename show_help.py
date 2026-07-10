#!/usr/bin/env python3

"""Print welcome message and quickstart hints."""


def main() -> None:
    wow = [
        "__        ______   __        __",
        "\\ \\      / / __ \\  \\ \\      / /",
        " \\ \\ /\\ / / |  | |  \\ \\ /\\ / / ",
        "  \\ V  V /| |__| |   \\ V  V /  ",
        "   \\_/\\_/  \\____/     \\_/\\_/   ",
    ]

    print("\n".join(wow))
    print()
    print("UFO is coming, start your training now!")
    print()
    print("Quick help:")
    print("- Training file: humanoidverse/train_mjlab.py")
    print("- Inference files:")
    print("  - humanoidverse/tracking_inference.py")
    print("  - humanoidverse/goal_inference.py")
    print("  - humanoidverse/reward_inference.py")


if __name__ == "__main__":
    main()
