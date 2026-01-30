# BananaTracker

Multi-object tracking using YOLOv8 detection + ByteTrack core (ported from McByte).

## Features

- YOLOv8-based object detection with class filtering
- ByteTrack-based tracking with Kalman filter motion prediction
- Camera motion compensation (ORB, ECC, SIFT, SparseOptFlow)
- Per-class visualization with customizable colors
- Support for special single-instance classes (e.g., puck in hockey)
- MOT format output for evaluation

## Installation

```bash
git clone https://github.com/USER/bananatracker.git
cd bananatracker
pip install -e .
```

## Quick Start

```python
from bananatracker import BananaTrackerConfig, BananaTrackerPipeline

config = BananaTrackerConfig(
    yolo_weights="path/to/weights.pt",
    class_names=["Player", "Puck", "Referee"],
    track_classes=[0, 1, 2],
    class_colors={
        "Player": (255, 0, 0),
        "Puck": (0, 255, 0),
        "Referee": (0, 0, 255),
    },
    output_video_path="output.mp4",
)

pipeline = BananaTrackerPipeline(config)
pipeline.process_video("input_video.mp4")
```

## Configuration

See `bananatracker/config.py` for all available configuration options.

## Architecture

BananaTracker uses a two-layer modular system:

1. **Layer 1 (Core Tracking)**: ByteTrack foundation with Kalman filter, IoU-based cost matrix, Hungarian matching, and camera motion compensation.

2. **Layer 2 (Future)**: Mask enhancement integration point for SAM + Cutie pipeline.

## License

MIT
