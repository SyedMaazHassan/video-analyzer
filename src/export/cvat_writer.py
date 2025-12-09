#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from lxml import etree
import json

def build_xml(phases_df, events_df, out_xml):
    root = etree.Element("annotations")
    meta = etree.SubElement(root, "meta")
    etree.SubElement(meta, "task")
    tracks_el = root

    track_id = 0
    for idx, r in phases_df.iterrows():
        t = etree.SubElement(tracks_el, "track", id=str(track_id), label=r["phase"])
        b1 = etree.SubElement(t, "box", frame=str(int(r["start_frame"])), outside="0", occluded="0", xtl="0", ytl="0", xbr="1", ybr="1")
        b2 = etree.SubElement(t, "box", frame=str(int(r["end_frame"] + 1)), outside="1", occluded="0", xtl="0", ytl="0", xbr="1", ybr="1")
        track_id += 1

    for idx, r in events_df.iterrows():
        t = etree.SubElement(tracks_el, "track", id=str(track_id), label=r["event"])
        b = etree.SubElement(t, "box", frame=str(int(r["frame"])), outside="0", occluded="0", xtl="0", ytl="0", xbr="2", ybr="2")
        try:
            attrs = json.loads(r.get("attributes", "{}"))
        except Exception:
            attrs = {}
        for k, v in attrs.items():
            a = etree.SubElement(b, "attribute", name=str(k))
            a.text = str(v)
        track_id += 1

    tree = etree.ElementTree(root)
    tree.write(str(out_xml), pretty_print=True, encoding="utf-8", xml_declaration=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=str, required=True)
    ap.add_argument("--out_xml", type=str, required=True)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    phases = pd.read_csv(pred_dir / "pred_phases.csv")
    events = (
        pd.read_csv(pred_dir / "pred_events.csv")
        if (pred_dir / "pred_events.csv").exists()
        else pd.DataFrame(columns=["case_id", "event", "frame", "attributes"])
    )

    build_xml(phases, events, Path(args.out_xml))
    print("Wrote CVAT XML:", args.out_xml)


if __name__ == "__main__":
    main()
