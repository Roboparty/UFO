# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mediapy
import numpy as np
import pandas as pd


def logfile_to_video_directory(logfile: str | Path):
    log_dir = Path(logfile).parent
    video_dir = log_dir / "videos"
    video_dir.mkdir(exist_ok=True)
    return video_dir

@dataclasses.dataclass
class CSVLogger:
    filename: Union[str, Path]
    fields: Optional[List[str]] = None

    def _ensure_fields(self, keys: Iterable[str]) -> None:
        path = Path(self.filename)
        has_data = path.exists() and path.stat().st_size > 0
        if self.fields is None:
            self.fields = list(pd.read_csv(path, nrows=0).columns) if has_data else sorted(keys)
        added = [key for key in sorted(keys) if key not in self.fields]
        if added:
            self.fields = list(self.fields) + added
            if has_data:
                pd.read_csv(path).reindex(columns=self.fields, fill_value="").to_csv(path, index=False)
                return
        if not has_data:
            pd.DataFrame(columns=self.fields).to_csv(path, index=False)

    def log(self, log_data: Dict[str, Any]) -> None:
        self._ensure_fields(log_data.keys())
        data = {field: log_data.get(field, "") for field in self.fields}  # Ensure all fields are present
        islist = [isinstance(v, Iterable) and not isinstance(v, str) for k, v in data.items()]
        if all(islist):
            df = pd.DataFrame(data)
        elif not any(islist):
            df = pd.DataFrame([data])
        else:
            raise RuntimeError("Fields should all be a numbers, a string or iterable objects. We don't support mixed types.")
        df.to_csv(self.filename, mode="a", header=False, index=False)

    def log_many(self, rows: Iterable[Dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        self._ensure_fields({key for row in rows for key in row})
        data = [{field: row.get(field, "") for field in self.fields} for row in rows]
        pd.DataFrame(data, columns=self.fields).to_csv(self.filename, mode="a", header=False, index=False)

    def log_video(self, filename: str, frames: list[np.ndarray], fps: int) -> None:
        # Implement video logging logic here
        output_path = logfile_to_video_directory(self.filename) / filename
        # breakpoint()  # use PYTHONBREAKPOINT=0 to disable, or install ipdb for a nicer debugger
        mediapy.write_video(output_path, frames, fps=fps)


@dataclasses.dataclass
class JSONLogger:
    filename: Union[str, Path]
    fields: Optional[List[str]] = None

    def log(self, log_data: Dict[str, Any]) -> None:
        if self.fields is None:
            self.fields = sorted(list(log_data.keys()))
            if not Path(self.filename).exists():
                with open(self.filename, "w+") as f:
                    json.dump({k: [] for k in self.fields}, f)

        # not the most efficient way of logging since we cannot append
        with open(self.filename, "r+") as f:
            logz = json.load(f)
        with open(self.filename, "w+") as f:
            for field in self.fields:
                logz[field].append(log_data.get(field, ""))
            json.dump(logz, f)
