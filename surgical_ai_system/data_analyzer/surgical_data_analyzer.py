import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json

def get_video_metadata(xml_path):
    """Finds and parses the video metadata file."""
    video_dir = Path(xml_path).parent
    for f in os.listdir(video_dir):
        if f.endswith('_metadata.txt'):
            with open(video_dir / f, 'r') as meta_file:
                metadata = json.load(meta_file)
                for stream in metadata.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        # r_frame_rate can be '30/1', so we evaluate it
                        fps = eval(stream.get('r_frame_rate', '30/1'))
                        total_frames = int(stream.get('nb_frames', 0))
                        duration_sec = float(stream.get('duration', 0))
                        return fps, total_frames, duration_sec
    return 30, 0, 0 # Default fps

def parse_annotation(xml_path):
    """
    Parses a single CVAT XML 1.1 annotation file to extract detailed phase and event data.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        print(f"Warning: Could not parse {xml_path}. Skipping.")
        return None, None

    fps, total_frames, duration_sec = get_video_metadata(xml_path)

    phases = []
    events = []

    for track in root.findall("track"):
        label = track.get("label")
        boxes = track.findall("box")
        if not boxes:
            continue

        start_frame = min([int(b.get("frame")) for b in boxes])
        end_frame = max([int(b.get("frame")) for b in boxes])
        
        attributes = {attr.get("name"): attr.text for attr in boxes[0].findall("attribute")}

        item = {
            "label": label,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_frames": end_frame - start_frame,
            "duration_sec": (end_frame - start_frame) / fps,
        }
        item.update(attributes)

        # Based on schema and guides, separate phases and events
        if label in ["Portal Placement", "Diagnostic Arthroscopy", "Labral Mobilization", 
                     "Glenoid Preparation", "Anchor Placement", "Suture Passage", 
                     "Suture Tensioning", "Final Inspection"]:
            phases.append(item)
        else:
            events.append(item)
            
    return pd.DataFrame(phases), pd.DataFrame(events)


def plot_phase_timeline(df_phases, output_path, video_filename):
    """Plots a Gantt-style chart for surgical phases."""
    if df_phases.empty:
        return
        
    df_phases = df_phases.sort_values("start_frame").reset_index()
    
    plt.figure(figsize=(12, 6))
    for i, phase in df_phases.iterrows():
        plt.barh(y=phase['label'], left=phase['start_frame'], width=phase['duration_frames'])
    
    plt.xlabel("Frame Number")
    plt.title(f"Surgical Phase Timeline for {video_filename}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_directory(dir_path, output_dir):
    """Analyzes all XML files in a directory and its subdirectories."""
    xml_files = list(Path(dir_path).rglob("*.xml"))
    print(f"Found {len(xml_files)} XML files to analyze.")

    all_phases_list = []
    all_events_list = []

    for xml_file in xml_files:
        video_filename = xml_file.stem
        print(f"Processing {video_filename}...")
        
        df_phases, df_events = parse_annotation(xml_file)

        if df_phases is None:
            continue

        # Per-surgery analysis
        video_output_dir = Path(output_dir) / "per_surgery_analysis" / video_filename
        video_output_dir.mkdir(parents=True, exist_ok=True)

        if not df_phases.empty:
            df_phases.to_csv(video_output_dir / "phases.csv", index=False)
            plot_phase_timeline(df_phases, video_output_dir / "phase_timeline.png", video_filename)
            all_phases_list.append(df_phases.assign(video=video_filename))

        if not df_events.empty:
            df_events.to_csv(video_output_dir / "events.csv", index=False)
            all_events_list.append(df_events.assign(video=video_filename))

    # Aggregate analysis
    if not all_phases_list:
        print("No phase data found to aggregate.")
        return

    df_all_phases = pd.concat(all_phases_list)
    df_all_events = pd.concat(all_events_list) if all_events_list else pd.DataFrame()

    agg_output_dir = Path(output_dir) / "aggregate_analysis"
    agg_output_dir.mkdir(exist_ok=True)

    df_all_phases.to_csv(agg_output_dir / "all_phases_data.csv", index=False)
    if not df_all_events.empty:
        df_all_events.to_csv(agg_output_dir / "all_events_data.csv", index=False)

    # Generate aggregate plots
    # Phase Duration Distribution
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df_all_phases, x="duration_sec", y="label")
    plt.title("Distribution of Phase Durations (seconds)")
    plt.tight_layout()
    plt.savefig(agg_output_dir / "agg_phase_duration_boxplot.png")
    plt.close()

    # Event Counts
    if not df_all_events.empty:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df_all_events, y="label", order=df_all_events['label'].value_counts().index)
        plt.title("Total Event Counts Across All Surgeries")
        plt.tight_layout()
        plt.savefig(agg_output_dir / "agg_event_counts.png")
        plt.close()

    print(f"\\nAnalysis complete. Results are in '{output_dir}'")


if __name__ == "__main__":
    # The user provided this path.
    # IMPORTANT: This path should contain the video folders, where each folder has an XML.
    # e.g., /home/anas/Downloads/video_samples/video_001/annotations.xml
    # e.g., /home/anas/Downloads/video_samples/video_002/annotations.xml
    dir_path = "/home/anas/Downloads/video_samples"
    
    # We'll save results in the project directory.
    output_dir = "analysis_results"
    
    # Create the main output directory
    Path(output_dir).mkdir(exist_ok=True)

    analyze_directory(dir_path, output_dir)
