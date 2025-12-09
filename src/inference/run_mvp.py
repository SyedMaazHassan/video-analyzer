#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", type=str, required=True)
    ap.add_argument("--tables_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    td = Path(args.tables_dir)
    outd = Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)

    phases = (
        pd.read_csv(td / "phases.csv")
        if (td / "phases.csv").exists()
        else pd.DataFrame(columns=["case_id", "phase", "start_frame", "end_frame"])
    )
    events = (
        pd.read_csv(td / "events.csv")
        if (td / "events.csv").exists()
        else pd.DataFrame(columns=["case_id", "event", "frame", "attributes"])
    )

    pred_phases = phases.copy()
    pred_events = events.copy()

    def frames_to_min(fr, fps):
        return fr / (fps * 60.0)

    pred_phases["duration_frames"] = pred_phases["end_frame"] - pred_phases["start_frame"] + 1
    pred_phases["duration_min"] = pred_phases["duration_frames"].apply(lambda x: frames_to_min(x, args.fps))

    case_summ = pred_phases.groupby(["case_id", "phase"]) ["duration_min"].sum().reset_index()
    total = (
        pred_phases.groupby("case_id")["duration_min"].sum().reset_index().rename(columns={"duration_min": "total_procedure_min"})
    )

    pred_phases.to_csv(outd / "pred_phases.csv", index=False)
    pred_events.to_csv(outd / "pred_events.csv", index=False)
    case_summ.to_csv(outd / "pred_phase_durations.csv", index=False)
    total.to_csv(outd / "pred_total.csv", index=False)

    print("Saved predictions to", outd)


if __name__ == "__main__":
    main()
