import os
import xml.etree.ElementTree as ET
from collections import Counter
import pandas as pd
from pathlib import Path

def find_xml_files(directory):
    """Recursively find all XML files in directory and subdirectories"""
    xml_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(root, file))
    return xml_files

def parse_annotation(xml_path):
    """Parse a single XML annotation file"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        instrument_counts = Counter()
        phase_counts = Counter()
        event_counts = Counter()

        # Track phases
        phases = [
            "Portal Placement", "Diagnostic Arthroscopy", "Labral Mobilization",
            "Glenoid Preparation", "Final Inspection", "Anchor Placement",
            "Suture Tensioning", "Suture Passage"
        ]

        for track in root.findall(".//track"):
            label = track.get("label")
            if label in phases:
                phase_counts[label] += 1

        # Count events from shapes
        for shape in root.findall(".//image/box") + root.findall(".//image/polygon"):
            label = shape.get("label")
            if label in ["Bleeding", "Suture Attempt", "Anchor Pullout", "Anchor Reposition"]:
                event_counts[label] += 1
            elif label in ["Instrument Entry", "Instrument Exit"]:
                for attr in shape.findall(".//attribute"):
                    if attr.get("name") == "Instrument":
                        instrument_counts[attr.text] += 1

        return instrument_counts, phase_counts, event_counts
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return Counter(), Counter(), Counter()

def analyze_annotations(dir_path):
    """Analyze all XML files in directory and subdirectories"""
    xml_files = find_xml_files(dir_path)
    total_files = len(xml_files)
    
    print(f"\n🔍 Found {total_files} XML files")
    
    all_instruments = Counter()
    all_phases = Counter()
    all_events = Counter()
    
    for xml_file in xml_files:
        print(f"Processing: {os.path.basename(xml_file)}")
        i, p, e = parse_annotation(xml_file)
        all_instruments.update(i)
        all_phases.update(p)
        all_events.update(e)

    # Convert to DataFrames
    instrument_df = pd.DataFrame(all_instruments.items(), columns=["Instrument", "Count"])
    phase_df = pd.DataFrame(all_phases.items(), columns=["Phase", "Count"])
    event_df = pd.DataFrame(all_events.items(), columns=["Event", "Count"])

    # Sort by count descending
    instrument_df = instrument_df.sort_values("Count", ascending=False)
    phase_df = phase_df.sort_values("Count", ascending=False)
    event_df = event_df.sort_values("Count", ascending=False)

    # Save results
    output_dir = Path(dir_path) / "analysis_results"
    output_dir.mkdir(exist_ok=True)
    
    instrument_df.to_csv(output_dir / "instrument_distribution.csv", index=False)
    phase_df.to_csv(output_dir / "phase_distribution.csv", index=False)
    event_df.to_csv(output_dir / "event_distribution.csv", index=False)

    print("\n📊 ANALYSIS RESULTS:")
    print("\nInstrument Distribution:")
    print(instrument_df)
    print("\nPhase Distribution:")
    print(phase_df)
    print("\nEvent Distribution:")
    print(event_df)
    print(f"\n💾 Results saved to: {output_dir}")

if __name__ == "__main__":
    dir_path = "/home/anas/Downloads/video_samples"
    analyze_annotations(dir_path)