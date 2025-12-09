#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables_dir", type=str, required=True)
    ap.add_argument("--videos_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--clip_len", type=int, default=64)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    td = Path(args.tables_dir)
    outd = Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)

    events = (
        pd.read_csv(td / "events.csv")
        if (td / "events.csv").exists()
        else pd.DataFrame(columns=["case_id", "event", "frame", "attributes"])
    )
    phases = (
        pd.read_csv(td / "phases.csv")
        if (td / "phases.csv").exists()
        else pd.DataFrame(columns=["case_id", "phase", "start_frame", "end_frame"])
    )

    rows = []

    for _, r in events.iterrows():
        start = max(0, int(r["frame"]) - args.clip_len // 2)
        end = start + args.clip_len - 1
        rows.append({
            "case_id": r["case_id"],
            "start_frame": start,
            "end_frame": end,
            "label": r["event"],
            "split": "train",
        })

    for _, pr in phases.iterrows():
        step = args.stride * args.clip_len
        cursor = int(pr["start_frame"])
        while cursor + args.clip_len <= pr["end_frame"]:
            rows.append({
                "case_id": pr["case_id"],
                "start_frame": cursor,
                "end_frame": cursor + args.clip_len - 1,
                "label": "NEGATIVE",
                "split": "train",
            })
            cursor += step

    man = pd.DataFrame(rows)
    man.to_csv(outd / "event_clip_manifest.csv", index=False)
    print("Saved clip manifest:", outd / "event_clip_manifest.csv")


if __name__ == "__main__":
    main()
