#!/usr/bin/env python3
"""
Publish 256D latent z over ZMQ (PUB), for wiring tests.

Message format: raw float32 array with shape (256,).
Subscriber should use CONFLATE=1 to always keep the latest z.
"""

import argparse
import time

import numpy as np
import zmq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Z latent publisher (test utility)")
    p.add_argument("--bind", type=str, default="tcp://*:28711", help="ZMQ bind address")
    p.add_argument("--hz", type=float, default=50.0, help="Publish rate")
    p.add_argument(
        "--mode",
        type=str,
        default="standing",
        choices=["standing", "random_walk", "from_npy"],
        help="standing: fixed z with ||z||=16; random_walk: smooth random; from_npy: replay npy rows",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--npy", type=str, default="", help="Path to .npy with shape (T,256) for from_npy")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(args.bind)

    dt = 1.0 / float(args.hz)
    rng = np.random.default_rng(int(args.seed))

    if args.mode == "standing":
        z = np.zeros(256, dtype=np.float32)
        z[0] = 16.0  # make ||z|| ~= 16 (matches common training normalization)
        stream = None
    elif args.mode == "random_walk":
        z = rng.standard_normal(256).astype(np.float32)
        z = z / (np.linalg.norm(z) + 1e-8) * 16.0
        stream = None
    else:
        if not args.npy:
            raise SystemExit("--npy is required for mode=from_npy")
        stream = np.load(args.npy).astype(np.float32)
        if stream.ndim != 2 or stream.shape[1] != 256:
            raise SystemExit(f"bad npy shape {stream.shape}, expected (T,256)")
        z = stream[0].copy()
        idx = 0

    print(f"[z_pub] bind={args.bind} hz={args.hz:.1f} mode={args.mode}")
    time.sleep(0.2)  # allow subscribers to connect (PUB slow-joiner)

    next_t = time.perf_counter()
    n = 0
    while True:
        now = time.perf_counter()
        if now < next_t:
            time.sleep(max(0.0, next_t - now))
        next_t += dt

        if args.mode == "random_walk":
            z = 0.98 * z + 0.02 * rng.standard_normal(256).astype(np.float32)
            z = z / (np.linalg.norm(z) + 1e-8) * 16.0
        elif args.mode == "from_npy":
            idx = (idx + 1) % int(stream.shape[0])
            z = stream[idx]

        try:
            sock.send(z.tobytes(), flags=zmq.DONTWAIT)
        except zmq.Again:
            pass

        n += 1
        if n % int(max(1, args.hz)) == 0:
            zn = float(np.linalg.norm(z))
            print(f"[z_pub] sent={n} ||z||={zn:.3f}")


if __name__ == "__main__":
    main()

