#!/usr/bin/env python3
"""Command-line interface for BananaTracker.

Usage:
    python run_tracking.py --weights model.pt --video input.mp4 --output output.mp4
"""

import argparse
import json
from bananatracker import BananaTrackerConfig, BananaTrackerPipeline


def parse_class_colors(colors_str: str) -> dict:
    """Parse class colors from JSON string.

    Example: '{"Player": [255, 0, 0], "Puck": [0, 255, 0]}'
    """
    if not colors_str:
        return {}
    try:
        colors_dict = json.loads(colors_str)
        return {k: tuple(v) for k, v in colors_dict.items()}
    except json.JSONDecodeError:
        print(f"Warning: Could not parse class_colors: {colors_str}")
        return {}


def parse_int_list(list_str: str) -> list:
    """Parse comma-separated integer list.

    Example: '0,1,2' -> [0, 1, 2]
    """
    if not list_str:
        return None
    try:
        return [int(x.strip()) for x in list_str.split(',')]
    except ValueError:
        print(f"Warning: Could not parse integer list: {list_str}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='BananaTracker: Multi-object tracking with YOLOv8 + ByteTrack'
    )

    # Required arguments
    parser.add_argument('--weights', '-w', type=str, required=True,
                        help='Path to YOLOv8 weights file (.pt)')
    parser.add_argument('--video', '-v', type=str, required=True,
                        help='Path to input video file')

    # Detection settings
    parser.add_argument('--class-names', type=str, default='',
                        help='Comma-separated class names (e.g., "Player,Puck,Referee")')
    parser.add_argument('--track-classes', type=str, default=None,
                        help='Comma-separated class indices to track (e.g., "0,1,2"). None = all')
    parser.add_argument('--special-classes', type=str, default=None,
                        help='Comma-separated class indices for max-conf-only filtering (e.g., "1")')
    parser.add_argument('--det-thresh', type=float, default=0.25,
                        help='Detection confidence threshold (default: 0.25)')

    # Tracker settings
    parser.add_argument('--track-thresh', type=float, default=0.6,
                        help='Track confidence threshold (default: 0.6)')
    parser.add_argument('--track-buffer', type=int, default=30,
                        help='Frames to keep lost tracks (default: 30)')
    parser.add_argument('--match-thresh', type=float, default=0.8,
                        help='Matching threshold (default: 0.8)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video FPS for buffer calculation (default: 30)')
    parser.add_argument('--cmc-method', type=str, default='orb',
                        choices=['orb', 'ecc', 'sift', 'sparseOptFlow', 'none'],
                        help='Camera motion compensation method (default: orb)')

    # Visualization settings
    parser.add_argument('--class-colors', type=str, default='',
                        help='JSON string for class colors (e.g., \'{"Player": [255,0,0]}\')')
    parser.add_argument('--no-track-id', action='store_true',
                        help='Hide track IDs in visualization')
    parser.add_argument('--line-thickness', type=int, default=2,
                        help='Bounding box line thickness (default: 2)')

    # Output settings
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to output video file')
    parser.add_argument('--output-txt', type=str, default=None,
                        help='Path to MOT format output file')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for inference (default: cuda:0)')

    args = parser.parse_args()

    # Build configuration
    class_names = [x.strip() for x in args.class_names.split(',')] if args.class_names else []

    config = BananaTrackerConfig(
        # Detection
        yolo_weights=args.weights,
        class_names=class_names,
        track_classes=parse_int_list(args.track_classes),
        special_classes=parse_int_list(args.special_classes),
        detection_conf_thresh=args.det_thresh,

        # Tracker
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        fps=args.fps,
        cmc_method=args.cmc_method,

        # Visualization
        class_colors=parse_class_colors(args.class_colors),
        show_track_id=not args.no_track_id,
        line_thickness=args.line_thickness,

        # Output
        output_video_path=args.output,
        output_txt_path=args.output_txt,
        device=args.device,
    )

    # Run tracking
    print(f"Loading model from: {args.weights}")
    print(f"Processing video: {args.video}")

    pipeline = BananaTrackerPipeline(config)
    all_tracks = pipeline.process_video(args.video)

    print(f"Processed {len(all_tracks)} frames")

    if args.output:
        print(f"Output video saved to: {args.output}")
    if args.output_txt:
        print(f"MOT results saved to: {args.output_txt}")


if __name__ == '__main__':
    main()
