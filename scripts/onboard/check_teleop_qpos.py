#!/usr/bin/env python3
"""Validate teleop bridge qpos replies without touching robot control."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import zmq


def _extract_qpos(payload: dict[str, Any]) -> np.ndarray | None:
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        return None
    frame = frames[-1]
    if not isinstance(frame, dict):
        return None

    root_pos = frame.get("root_pos")
    root_quat = frame.get("root_quat")
    dof_pos = frame.get("dof_pos")
    if not (isinstance(root_pos, list) and isinstance(root_quat, list) and isinstance(dof_pos, list)):
        return None
    if len(root_pos) != 3 or len(root_quat) != 4 or len(dof_pos) != 29:
        return None

    try:
        qpos = np.asarray(root_pos + root_quat + dof_pos, dtype=np.float32)
    except Exception:
        return None
    if qpos.shape != (36,) or not np.all(np.isfinite(qpos)):
        return None
    qnorm = float(np.linalg.norm(qpos[3:7]))
    if not np.isfinite(qnorm) or qnorm < 1e-6:
        return None
    return qpos


def main() -> int:
    parser = argparse.ArgumentParser(description="Check teleop pose bridge qpos replies")
    parser.add_argument("--req-addr", default="tcp://127.0.0.1:28701")
    parser.add_argument("--rep-addr", default="tcp://127.0.0.1:28702")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--period", type=float, default=0.02)
    parser.add_argument("--timeout-ms", type=int, default=1000)
    parser.add_argument("--min-valid", type=int, default=1)
    parser.add_argument("--max-retarget-age-ms", type=float, default=250.0)
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="accept fallback/static qpos replies without live retarget_age_ms",
    )
    args = parser.parse_args()

    ctx = zmq.Context.instance()
    req = ctx.socket(zmq.PUSH)
    req.setsockopt(zmq.LINGER, 0)
    req.connect(str(args.req_addr))

    rep = ctx.socket(zmq.PULL)
    rep.setsockopt(zmq.LINGER, 0)
    rep.connect(str(args.rep_addr))

    poller = zmq.Poller()
    poller.register(rep, zmq.POLLIN)

    sent = 0
    replies = 0
    valid = 0
    fallback = 0
    invalid = 0
    retarget_ages: list[float] = []
    qpos_rows: list[np.ndarray] = []
    recv_times: list[float] = []

    deadline = time.monotonic() + max(0.0, float(args.duration))
    next_send = time.monotonic()
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_send:
            req.send_string(json.dumps({"start": sent == 0, "t_req_ms": int(time.time() * 1000)}))
            sent += 1
            next_send = now + max(0.001, float(args.period))

        events = dict(poller.poll(int(args.timeout_ms)))
        if rep not in events:
            continue

        recv_times.append(time.monotonic())
        replies += 1
        try:
            payload = json.loads(rep.recv_string())
        except Exception:
            invalid += 1
            continue
        if not isinstance(payload, dict):
            invalid += 1
            continue

        if bool(payload.get("no_interp_applied")):
            fallback += 1
        age = payload.get("retarget_age_ms")
        if age is not None:
            try:
                retarget_ages.append(float(age))
            except Exception:
                pass

        qpos = _extract_qpos(payload)
        if qpos is None:
            invalid += 1
            continue
        qpos_rows.append(qpos)
        valid += 1

    print(f"[INFO] req_addr: {args.req_addr}")
    print(f"[INFO] rep_addr: {args.rep_addr}")
    print(f"[INFO] sent: {sent}")
    print(f"[INFO] replies: {replies}")
    print(f"[INFO] valid_qpos: {valid}")
    print(f"[INFO] invalid_or_empty: {invalid}")
    print(f"[INFO] fallback_replies: {fallback}")

    if retarget_ages:
        print(f"[INFO] retarget_age_ms_min: {float(np.min(retarget_ages))}")
        print(f"[INFO] retarget_age_ms_p95: {float(np.percentile(retarget_ages, 95))}")
        print(f"[INFO] retarget_age_ms_max: {float(np.max(retarget_ages))}")

    if qpos_rows:
        qpos = np.stack(qpos_rows)
        quat_norm = np.linalg.norm(qpos[:, 3:7], axis=1)
        root_z = qpos[:, 2]
        print("[INFO] qpos_size: 36")
        print(f"[INFO] finite: {bool(np.isfinite(qpos).all())}")
        print(f"[INFO] root_z_min: {float(root_z.min())}")
        print(f"[INFO] root_z_max: {float(root_z.max())}")
        print(f"[INFO] quat_norm_min: {float(quat_norm.min())}")
        print(f"[INFO] quat_norm_max: {float(quat_norm.max())}")
        if len(recv_times) > 1:
            intervals_ms = np.diff(np.asarray(recv_times)) * 1000.0
            print(f"[INFO] reply_interval_ms_p50: {float(np.percentile(intervals_ms, 50))}")
            print(f"[INFO] reply_interval_ms_p95: {float(np.percentile(intervals_ms, 95))}")

    if valid < int(args.min_valid):
        print(f"[FAIL] expected at least {args.min_valid} valid qpos frame(s)")
        return 1
    if not bool(args.allow_fallback):
        if fallback > 0:
            print(f"[FAIL] {fallback} fallback reply/replies observed; live retarget required")
            return 1
        if not retarget_ages:
            print("[FAIL] no retarget_age_ms values observed; live retarget required")
            return 1
    if retarget_ages and float(args.max_retarget_age_ms) >= 0:
        too_old = [x for x in retarget_ages if x > float(args.max_retarget_age_ms)]
        if too_old:
            print(f"[FAIL] {len(too_old)} retarget frame(s) older than {args.max_retarget_age_ms} ms")
            return 1

    print("[OK] teleop bridge returned finite 36-D qpos frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
