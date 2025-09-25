# surgical_ai_system/core/data_processing.py

import xml.etree.ElementTree as ET
import json
from pathlib import Path
import pandas as pd
import cv2 # Using OpenCV for video processing
from typing import List, Tuple

from surgical_ai_system.config.system_config import app_config
from surgical_ai_system.core.data_structures.surgical_entities import SurgicalPhase, SurgicalEvent, SurgicalCaseData

def get_video_metadata(video_path: Path) -> Tuple[float, int]:
    """
    Reads video file to get FPS and total frame count using OpenCV.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames

def parse_cvat_xml(xml_path: Path, case_id: str) -> Tuple[List[SurgicalPhase], List[SurgicalEvent]]:
    """
    Parses a CVAT XML 1.1 file and returns lists of phase and event objects.
    This is an evolution of the parser from the analysis script.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        print(f"Warning: Could not parse {xml_path}. Skipping.")
        return [], []

    phases = []
    events = []
    
    phase_labels = set(app_config.labels['phases'])
    event_labels = set(app_config.labels['events'])

    for track in root.findall("track"):
        label = track.get("label")
        boxes = track.findall("box")
        if not boxes:
            continue

        attributes = {attr.get("name"): attr.text for attr in boxes[0].findall("attribute")}

        if label in phase_labels:
            start_frame = min([int(b.get("frame")) for b in boxes])
            end_frame = max([int(b.get("frame")) for b in boxes])
            phases.append(SurgicalPhase(case_id, label, start_frame, end_frame))
        
        elif label in event_labels:
            # For events, we take the start frame of the box as the event time
            frame = int(boxes[0].get("frame"))
            events.append(SurgicalEvent(case_id, label, frame, attributes))
            
    return phases, events

def load_case_data(case_dir: Path) -> SurgicalCaseData:
    """
    Loads all data for a single case (video metadata, annotations).
    """
    case_id = case_dir.name
    
    # Find video and XML files
    try:
        video_path = next(case_dir.glob("*.mp4"))
        xml_path = next(case_dir.glob("*.xml"))
    except StopIteration:
        raise FileNotFoundError(f"Could not find required video (.mp4) or annotation (.xml) file in {case_dir}")

    fps, total_frames = get_video_metadata(video_path)
    phases, events = parse_cvat_xml(xml_path, case_id)
    
    return SurgicalCaseData(
        case_id=case_id,
        video_path=str(video_path),
        fps=fps,
        total_frames=total_frames,
        phases=phases,
        events=events
    )

def extract_frames(video_path: str, frame_indices: List[int]):
    """
    Generator function to extract specific frames from a video file.
    This is more memory-efficient than loading all frames at once.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    current_frame_idx = 0
    target_indices = sorted(list(set(frame_indices)))
    target_idx_ptr = 0

    while cap.isOpened() and target_idx_ptr < len(target_indices):
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame_idx == target_indices[target_idx_ptr]:
            # Convert color from BGR (OpenCV default) to RGB
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            target_idx_ptr += 1
            
        current_frame_idx += 1
        
    cap.release()
