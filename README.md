# beeTracker

Tracks bee movement in a behavioral box using a fine-tuned YOLO11n model with BoT-SORT tracking.

## How It Works

1. **YOLO11n** detects bees in each frame and outputs bounding boxes
2. **BoT-SORT** assigns consistent IDs to each bee across frames, handling occlusion and overlap
3. **Telemetry engine** calculates speed, idle time, and position from tracked coordinates

This replaces the previous OpenCV threshold-based detection, which struggled with bees near edges, near the center food blob, and when bees overlapped. The YOLO model learns what a bee looks like rather than relying on pixel intensity, so it works reliably across lighting conditions and positions.

## Outputs

| File | Description |
|------|-------------|
| `bee_telemetry.csv` | Per-frame bee positions and speeds |
| `bee_summary_statistics.xlsx` | Avg/max speed, frames tracked, idle % per bee |
| `tracked_video.mp4` | Annotated video with bounding boxes, IDs, and trails |
| `speed_plot.png` | Speed over time graph for all bees |

## New Setup

Install UV https://docs.astral.sh/uv/

```bash
uv sync
```

## New Usage

```bash
uv run reduced_live_tracker.py --video test.mp4
```

## Old Setup

Requires Python 3.9–3.12.

```bash
make setup
```

This creates a virtual environment and installs all dependencies.

## Old Usage

Place your video and trained model in the project directory, then:

```bash
make run
```

or for live tracking from a camera feed:

```bash
make run-live
```

To clean up generated files and the virtual environment:

```bash
make clean
```