#!/usr/bin/env python3
"""Subscribe to a realtime z stream and validate float32 packet shape."""

from __future__ import annotations

import argparse
import time

import numpy as np
import zmq


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a UFO realtime z ZMQ stream")
    parser.add_argument("--addr", default="tcp://127.0.0.1:28711")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--timeout-ms", type=int, default=500)
    parser.add_argument("--expect-size", type=int, default=256)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--min-rate-hz", type=float, default=0.0)
    args = parser.parse_args()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(str(args.addr))

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    arrays: list[np.ndarray] = []
    recv_times: list[float] = []
    deadline = time.monotonic() + max(0.0, float(args.duration))
    while time.monotonic() < deadline:
        events = dict(poller.poll(int(args.timeout_ms)))
        if sock not in events:
            continue
        payload = sock.recv()
        arrays.append(np.frombuffer(payload, dtype=np.float32).copy())
        recv_times.append(time.monotonic())

    count = len(arrays)
    print(f"[INFO] addr: {args.addr}")
    print(f"[INFO] count: {count}")
    if count == 0:
        print("[FAIL] no z packets received")
        return 1

    sizes = sorted({int(x.size) for x in arrays})
    print(f"[INFO] sizes: {sizes}")
    if sizes != [int(args.expect_size)]:
        print(f"[FAIL] expected only size {args.expect_size}, got {sizes}")
        return 1

    z = np.stack(arrays)
    finite = bool(np.isfinite(z).all())
    print(f"[INFO] finite: {finite}")
    if not finite:
        print("[FAIL] z stream contains non-finite values")
        return 1

    norms = np.linalg.norm(z, axis=1)
    print(f"[INFO] norm_min: {float(norms.min())}")
    print(f"[INFO] norm_max: {float(norms.max())}")
    print(f"[INFO] norm_last: {float(norms[-1])}")

    if len(arrays) > 1:
        intervals_ms = np.diff(np.asarray(recv_times)) * 1000.0
        dz = np.max(np.abs(np.diff(z, axis=0)), axis=1)
        elapsed = max(1e-9, recv_times[-1] - recv_times[0])
        rate_hz = (len(arrays) - 1) / elapsed
        print(f"[INFO] rate_hz: {float(rate_hz)}")
        print(f"[INFO] interval_ms_p50: {float(np.percentile(intervals_ms, 50))}")
        print(f"[INFO] interval_ms_p95: {float(np.percentile(intervals_ms, 95))}")
        print(f"[INFO] max_abs_dz: {float(dz.max())}")
    else:
        rate_hz = 0.0
        print("[INFO] rate_hz: 0.0")
        print("[INFO] max_abs_dz: 0.0")

    if count < int(args.min_count):
        print(f"[FAIL] expected at least {args.min_count} packets")
        return 1
    if float(args.min_rate_hz) > 0.0 and rate_hz < float(args.min_rate_hz):
        print(f"[FAIL] expected rate >= {args.min_rate_hz} Hz")
        return 1

    print("[OK] z stream is finite and has expected shape")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
