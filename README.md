# beeTracker

Real-time bee tracking system. A Raspberry Pi captures video of a behavioral box and streams it over the network; an external device with an NVIDIA GPU runs detection and tracking, writes telemetry to Supabase, and publishes the annotated video as a public web dashboard.

## Purpose

SynBeeGee aims to protect bees from the inside out by engineering their gut microbiome. We focus on Snodgrassella Alvi, and introduce the organophosphate degradation gene. This gene encodes an enzyme that hydrolyzes toxic pesticides into harmless byproducts, thereby neutralizing their effects. Unlike external treatments, engineered S. Alvi can persist in the gut, spread throughout the hive, and be passed on to offspring microbiomes, providing long-term, colony-wide protection. This project extends synthetic biology beyond genetic engineering by creating an AI-enabled measurement and analysis database for honeybee colonies. By integrating hardware, software, and biological systems, our work aligns strongly with iGEM’s Measurement, Software, and Hardware villages, contributing tools and infrastructure that make complex biological behaviors quantifiable, reproducible, and actionable.
This repository contains the code for the real-time bee tracking system, which is a critical component of our project. By accurately tracking bee behavior and health metrics, we can assess the impact of our engineered S. Alvi on colony well-being and pesticide resistance. The data collected through this system will inform our understanding of how microbiome engineering affects bees at both individual and colony levels, ultimately guiding future iterations of our design and contributing to the broader field of synthetic biology.

## Architecture

Raspberry Pi: camera capture → RTSP stream (mediamtx + libcamera-vid)

Laptop: RTSP input → YOLOv11n detection → BoT-SORT tracking → telemetry → NVENC encoding → RTSP output (mediamtx) → HLS dashboard

## How it works

YOLO11n detects bees in each frame and outputs bounding boxes. BoT-SORT assigns consistent IDs across frames, handling occlusion and overlap. The telemetry engine derives speed (using wall-clock dt, since frames may be skipped under load), idle time, and trajectories from the tracked coordinates. Annotated frames are encoded with NVIDIA NVENC and re-published as RTSP, which mediamtx then serves as HLS for browser viewing.

## Hardware required

- **Raspberry Pi** with an IR camera module
- **Device with NVIDIA GPU** for running the tracker and HLS server

## Pi setup

Use mediamtx to capture the camera feed and publish it as RTSP.

## Laptop setup

Install [uv](https://docs.astral.sh/uv/) and sync the project:

```bash
uv sync
```

Modify `pyproject.toml` based on GPU.

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]
torchvision = [{ index = "pytorch-cu128" }]
```

Then `uv sync` again. Verify CUDA works:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Configuration

Create a `.env` in the project root:

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-or-service-key
RTSP_URL=rtsp://PI_IP:8554/cam
PUBLISH_URL=rtsp://localhost:8554/processed
```

Supabase schema is documented in the docstring at the top of `external_tracker.py`. Run those `CREATE TABLE` statements once in your Supabase SQL editor.

The default model is `bee.pt`. Swap it via the `MODEL_PATH` constant in the script.

## Running

Start things in this sequence:

```bash
# 1. Pi: already running mediamtx + libcamera-vid

# 2. External terminal 1: local mediamtx (RTSP receiver + HLS server)
./mediamtx

# 3. External terminal 2: tracker
uv run external_tracker.py

```

Useful flags on the tracker:

```bash
uv run external_tracker.py --no-display      # headless
uv run external_tracker.py --no-publish      # skip the HLS pipeline
uv run external_tracker.py --no-nvenc        # CPU encoding instead of NVENC
uv run external_tracker.py --video test.mp4  # offline video file instead of RTSP
```

## Outputs

| Output | Where | Content |
|--------|-------|---------|
| `output/telemetry_YYYYMMDD_HHMMSS.csv` | Local laptop | Per-frame: timestamp, frame, bee_id, x, y, speed_mps, idle |
| `bee_frame_observation` | Supabase table | Same as CSV, only tracked bees (track_id ≥ 0) |
| `bees_summary_per_minute` | Supabase table | Per-minute aggregates: avg_speed, mean_huddling, avg_temp |
| HLS stream | `https://YOUR-URL/processed/index.m3u8` | Live annotated video with boxes + trails |

The CSV writes everything (including untracked detections, `bee_id = -1`); the Supabase table only gets tracked bees because of the foreign key on `bees(bee_id)`. Old `bee_frame_observation` rows are deleted automatically every hour (>2 hours old).

## Legacy usage

The original single-machine workflow with `make` targets still works for offline analysis on a video file:

```bash
make setup    # creates venv, installs deps
make run      # runs tracker on a video file
make run-live # runs tracker on a local camera
make clean    # removes venv and outputs
```

This path produces additional analysis outputs (`bee_summary_statistics.xlsx`, `tracked_video.mp4`, `speed_plot.png`) via the older `tracker.py` script. Not used in the live RTSP pipeline.

## Project layout

```
beeTracker/
├── external_tracker.py     # main: live RTSP → YOLO → Supabase
├── demo.py                 # version with HLS publishing via mediamtx
├── live_tracker.py         # earlier version (pre-publishing)
├── tracker.py              # offline video analysis (legacy make targets)
├── reduced_live_tracker.py # stripped-down variant for full deployment on Pi (failure due current constraints)
├── bee.pt                  # current model (default)
├── best.pt, model.pt, new_model.pt  # earlier models
├── *_ncnn_model/           # NCNN exports for the Pi (not used on laptop)
├── mediamtx.yml            # external device's mediamtx config
├── botsort.yaml            # tracker config
├── pyproject.toml          # uv-managed dependencies
├── makefile                # legacy non-uv workflow
└── output/                 # CSV telemetry backups
```

## Future improvements

- Add a web dashboard with historical playback and telemetry graphs (instead of just the HLS video).
- Upgrade raspberry Pi or move to cloud compute for future contained experiments.
- Advanced analytics: behavior classification, anomaly detection, etc.
- Optimize the model and tracking for better accuracy and speed.

## Tools used for training the model

https://github.com/jonathanrandall/yolo_labelling_tool

## mediamtx.yml

logLevel: info

rtspAddress: :8554

hls: yes
hlsAddress: :8888
hlsAlwaysRemux: yes
hlsSegmentCount: 7
hlsSegmentDuration: 1s
hlsAllowOrigins: ['*']

rtmp: no
webrtc: no
srt: no

paths:
  processed:
    source: publisher



output goes to `http://localhost:8888/processed/index.m3u8`
