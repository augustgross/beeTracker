# beeTracker

Real-time bee tracking. A Raspberry Pi films a behavioral box and streams the video over RTSP. A second machine with an NVIDIA GPU pulls that stream, detects and tracks the bees, and writes telemetry to a local CSV and to Supabase. It can also re-publish the annotated video so it can be watched in a browser.

There are two entry points, and they are easy to mix up:

| Script | What it does |
|--------|--------------|
| `external_tracker.py` | RTSP in, YOLO + BoT-SORT, CSV + Supabase out. No video output. |
| `demo.py` | All of the above, plus it encodes the annotated frames with ffmpeg and pushes them to a local mediamtx, which serves them as HLS. |

The tracking and telemetry code is the same in both files. `demo.py --no-publish` behaves like `external_tracker.py`.

## Purpose

SynBeeGee is an iGEM project that tries to protect honeybees from pesticides by engineering their gut microbiome. We work with *Snodgrassella alvi* and give it an organophosphate degradation gene. The enzyme that gene encodes hydrolyzes organophosphate pesticides into harmless byproducts. Unlike an external treatment, engineered *S. alvi* can persist in the gut, spread through the hive, and be passed on to the next generation's microbiome, so the protection is long-term and colony-wide.

This repository is the measurement side of the project: an AI-based tracking system and database for bee behavior, which puts it in iGEM's Measurement, Software and Hardware villages. It records where each bee is, how fast it moves and how long it sits idle, so we can see what the engineered *S. alvi* does to colony well-being and pesticide resistance.

## Architecture

```
Raspberry Pi
  camera -> libcamera-vid -> mediamtx -> rtsp://PI_IP:8554/cam

Tracking machine (both scripts)
  RTSP in -> YOLO11 detection -> BoT-SORT tracking -> speed / idle / trails
          -> output/telemetry_*.csv
          -> Supabase (bee_frame_observation, bees_summary_per_minute)

Tracking machine (demo.py only)
  annotated frames -> ffmpeg (h264_nvenc or libx264) -> rtsp://localhost:8554/processed
                   -> mediamtx -> HLS at http://localhost:8888/processed/index.m3u8
```

## How it works

YOLO11 finds the bees in each frame. The default model, `bee.pt`, is a single-class YOLO11m trained at 1024 px, and the scripts run it at `imgsz=1024`. BoT-SORT, the Ultralytics default tracker, gives each bee an ID that persists across frames.

A background thread (`LatestFrameGrabber`) keeps only the newest frame from the stream. When inference is slower than the camera, old frames are dropped instead of queued, so the tracker never falls behind real time. Because frames get skipped, speed is computed from the wall-clock time between processed frames, not from the nominal frame rate.

Speed is the pixel distance a bee moved, converted to meters with `DISH_DIAMETER_METERS` (0.1) and `DISH_DIAMETER_PIXELS` (920), divided by that time step. A bee counts as idle below `IDLE_SPEED_THRESHOLD` (0.01 m/s). These constants sit at the top of each script. Re-measure the pixel diameter if the camera, its resolution or the dish changes.

Rows go to Supabase in batches of 300. Two background threads also talk to the database. One rebuilds the last two minutes of `bees_summary_per_minute` every 60 seconds. The other deletes `bee_frame_observation` rows older than two hours, once at startup and then once an hour.

In `demo.py` the annotated frames are piped to an ffmpeg subprocess. The publisher keeps only the latest frame too, so a slow encoder drops frames and never stalls inference. Every 500 frames the log shows how many frames were published and how many were dropped.

## Hardware

- Raspberry Pi with an IR camera module
- A machine with an NVIDIA GPU to run the tracker. It runs without one, slowly (see "Running without a GPU").
- For `demo.py`: `ffmpeg` on the PATH, built with `h264_nvenc` if you want NVENC, and a [mediamtx](https://github.com/bluenviron/mediamtx/releases) binary

## Pi setup

Use mediamtx to capture the camera feed and publish it as RTSP at `rtsp://PI_IP:8554/cam`.

## Tracking machine setup

Install [uv](https://docs.astral.sh/uv/) and sync the project:

```bash
uv sync
```

### Getting a GPU build of torch

`uv.lock` resolves torch 2.11.0 from PyPI, and what PyPI gives you depends on the OS:

- Linux: a CUDA 13.0 build. It needs NVIDIA driver 580 or newer. With an older driver `torch.cuda.is_available()` returns False.
- Windows: a CPU-only build. The GPU is never used.

To choose the CUDA build yourself, make torch and torchvision direct dependencies:

```toml
[project]
dependencies = [
    # ...existing entries...
    "torch",
    "torchvision",
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }]
torchvision = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }]
```

Use the index that matches your driver (`cu128`, `cu130`, ...). Then re-lock and check:

```bash
uv lock && uv sync
grep -A2 '^name = "torch"' uv.lock   # source should be download.pytorch.org, not pypi.org
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Checking which device you got

Both scripts choose the device with `"cuda" if torch.cuda.is_available() else "cpu"` and print one of these lines at startup:

```
Model loaded on cuda: NVIDIA GeForce RTX ...
Model loaded on cpu: CPU
```

There is no warning when it falls back to the CPU, so look for that line. A virtual machine only sees the GPU if the host passes it through. A default VirtualBox or Vagrant box does not.

## Running without a GPU

It runs, but slowly. These numbers are from one CPU core with the locked versions (torch 2.11.0, ultralytics 8.4.37) and a 1920x1080 frame:

| Model | imgsz | half | Speed |
|-------|-------|------|-------|
| `bee.pt` (YOLO11m) | 1024 | True, which is what the scripts do today | 0.7 FPS |
| `bee.pt` | 1024 | False | 1.3 FPS |
| `bee.pt` | 640 | False | 3.3 FPS |
| `best.pt` (YOLO11n) | 640 | False | 16 FPS |

Three things to know before trusting CPU output:

- The scripts pass `half=True` unconditionally and Ultralytics does not switch FP16 off on the CPU, so CPU inference runs in FP16 at about half the speed of FP32. Changing the call to `half=(device == "cuda")` fixes that.
- Below 1 FPS the speed column is wrong. When more than a second passes between frames the scripts treat it as a stall and substitute the nominal frame interval (1/20 s for a live stream) for the time step. That inflates speed by 20x or more.
- At a few FPS the bees move a long way between frames, so IDs switch more often and speeds get noisy.

The CPU is fine for checking that the pipeline works end to end. For real data use a GPU. If that is not possible, `best.pt` at 640 is fast enough, but it is an earlier and smaller model, so check its detections before relying on it.

## Configuration

Create a `.env` in the project root:

```
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_KEY=<service-role (secret) key>
RTSP_URL=rtsp://PI_IP:8554/cam
PUBLISH_URL=rtsp://localhost:8554/processed
```

`PUBLISH_URL` is only read by `demo.py`. `SUPABASE_URL` is the project URL shown in the API settings of the Supabase dashboard, not `https://supabase.com`.

The scripts print `Connected to Supabase` as soon as the client object exists, before any request is sent, so that line does not prove the URL or key is right. A wrong value shows up later as `Supabase upload error` when the first batch goes out. If either variable is missing, the scripts write the CSV only.

The model is set by the `MODEL_PATH` constant at the top of each script (default `bee.pt`), with `INFERENCE_IMGSZ` next to it.

### Supabase schema

The original `CREATE TABLE` statements are in the docstring at the top of `live_tracker.py`. The version below is the same except that `observation_id` is an identity column. The scripts never send an `observation_id`, so without a default every insert is rejected.

```sql
CREATE TABLE bees (
    bee_id INTEGER PRIMARY KEY
);

CREATE TABLE bee_frame_observation (
    observation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    bee_id INTEGER NOT NULL REFERENCES bees(bee_id),
    frame INTEGER NOT NULL,
    time_stamp TIMESTAMPTZ NOT NULL,
    date DATE GENERATED ALWAYS AS ((time_stamp AT TIME ZONE 'UTC')::date) STORED,
    x_coord DOUBLE PRECISION,
    y_coord DOUBLE PRECISION,
    speed DOUBLE PRECISION
);

CREATE INDEX idx_bfo_timestamp ON bee_frame_observation(time_stamp);
CREATE INDEX idx_bfo_bee_timestamp ON bee_frame_observation(bee_id, time_stamp);

CREATE TABLE bees_summary_per_minute (
    bee_id INTEGER NOT NULL REFERENCES bees(bee_id),
    minute_bucket TIMESTAMPTZ NOT NULL,
    avg_speed DOUBLE PRECISION,
    mean_huddling DOUBLE PRECISION,
    avg_temp DOUBLE PRECISION,
    PRIMARY KEY (bee_id, minute_bucket)
);

-- In the original schema; no script writes to it yet.
CREATE TABLE colony_summary_hourly (
    hour_bucket TIMESTAMPTZ PRIMARY KEY,
    avg_colony_speed DOUBLE PRECISION,
    total_huddle_clusters INTEGER,
    avg_cluster_size DOUBLE PRECISION
);
```

Two more things the scripts need from the database, neither of which they set up themselves.

Every `bee_id` has to exist in `bees` before an observation can reference it, and the scripts never insert into `bees`. A batch is one insert, so a single unknown ID fails all 300 rows. Either pre-fill the table or add a trigger that creates the row:

```sql
INSERT INTO bees (bee_id) SELECT generate_series(1, 100000) ON CONFLICT DO NOTHING;
```

The summary and cleanup threads call a Postgres function named `execute_sql` through `db.rpc(...)`. It has to exist, and because it runs whatever SQL it is given, it must not be callable with the anon key. A minimal version:

```sql
CREATE OR REPLACE FUNCTION execute_sql(sql text)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    EXECUTE sql;
END;
$$;

REVOKE EXECUTE ON FUNCTION execute_sql(text) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION execute_sql(text) TO service_role;
```

That is why `.env` uses the service-role (secret) key. Keep that key on the tracking machine. The anon key is the one that belongs in a browser dashboard.

## Running

```bash
# 1. Pi: mediamtx + libcamera-vid already streaming

# 2a. Telemetry only
uv run external_tracker.py

# 2b. Telemetry + browser stream. Start mediamtx first, in its own terminal.
./mediamtx
uv run demo.py
```

Stop with Esc in the video window, or Ctrl+C.

| Flag | `external_tracker.py` | `demo.py` | Effect |
|------|:---:|:---:|--------|
| `--rtsp URL` | yes | yes | Use this stream instead of `RTSP_URL` |
| `--video FILE` | yes | yes | Read a video file instead of RTSP |
| `--no-display` | yes | yes | No OpenCV window. Needed on a headless machine or over SSH. |
| `--publish-url URL` | no | yes | Use this instead of `PUBLISH_URL` |
| `--no-publish` | no | yes | Skip ffmpeg and HLS |
| `--no-nvenc` | no | yes | Encode with libx264 on the CPU |

`demo.py` only uses NVENC when CUDA is available and falls back to libx264 by itself otherwise. `--no-nvenc` is for machines where CUDA works but ffmpeg was built without `h264_nvenc`. If ffmpeg is not installed, `demo.py` says so and carries on without publishing. If mediamtx is not running, ffmpeg exits, the log shows `ffmpeg pipe died`, and tracking continues without the stream.

## mediamtx config

Only `demo.py` needs this. `mediamtx.yml` is not in the repo, so save the following next to the mediamtx binary:

```yaml
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
```

The stream is then at `http://localhost:8888/processed/index.m3u8`. To watch it from outside the local network, expose port 8888 through a tunnel. `demo.py` was written with Cloudflare Tunnel in mind.

## Outputs

| Output | Written by | Content |
|--------|------------|---------|
| `output/telemetry_YYYYMMDD_HHMMSS.csv` | both | timestamp, frame, bee_id, x, y, speed_mps, idle. Includes untracked detections as `bee_id = -1`. |
| `bee_frame_observation` (Supabase) | both | bee_id, frame, time_stamp, x_coord, y_coord, speed. Tracked bees only, and no idle column. |
| `bees_summary_per_minute` (Supabase) | both | `avg_speed` per bee per minute. `mean_huddling` and `avg_temp` are written as NULL for now. |
| HLS stream | `demo.py` | Annotated video with boxes, IDs, trails and an FPS counter |

Untracked detections stay out of Supabase because `bee_id = -1` would break the foreign key on `bees(bee_id)`.

A `bee_id` is a track, not an individual bee. IDs start again at 1 every time the script starts, and a bee that is lost for longer than the tracker's buffer comes back under a new ID. Keep that in mind when comparing per-bee numbers across runs.

## Legacy scripts

The original single-machine workflow still works for offline analysis:

```bash
make setup    # creates venv, installs requirements.txt
make run      # tracker.py on a video file
make run-live # live_tracker.py on local camera 0 (run make setup first)
make clean    # removes venv and outputs
```

`tracker.py` reads the video and model paths from constants at the top of the file (`new_video.mp4`, `new_model.pt`). It writes `bee_telemetry.csv`, `bee_summary_statistics.xlsx`, `tracked_video.mp4` and `speed_plot.png` to `output/`. None of this is used by the live RTSP pipeline.

## Project layout

```
beeTracker/
├── external_tracker.py      # live RTSP -> YOLO -> CSV + Supabase
├── demo.py                  # same, plus ffmpeg/mediamtx HLS publishing
├── live_tracker.py          # earlier local-camera version; its docstring holds the Supabase schema
├── tracker.py               # offline video analysis (make run)
├── reduced_live_tracker.py  # NCNN variant meant to run on the Pi itself; too slow on the current Pi
├── bee.pt                   # default model: YOLO11m, one class, trained at 1024
├── best.pt                  # YOLO11n trained at 640, used by live_tracker.py
├── model.pt, new_model.pt   # YOLO11s trained at 480; identical files, tracker.py uses new_model.pt
├── best_ncnn_model/, model_ncnn_model/   # NCNN exports for the Pi
├── botsort.yaml             # tuned BoT-SORT settings, not loaded by any script yet
├── pyproject.toml, uv.lock  # uv-managed dependencies
├── requirements.txt, makefile   # legacy non-uv workflow
└── output/                  # CSV telemetry (git-ignored)
```

## Known issues

- `half=True` is passed on the CPU as well. See "Running without a GPU".
- The time-step guard breaks speed values below 1 FPS. Same section.
- `botsort.yaml` is never passed to `model.track()`, so tracking runs on the Ultralytics default BoT-SORT settings, which are quite different (`match_thresh` 0.8 instead of 0.99, `track_buffer` 30 instead of 60, `new_track_thresh` 0.25 instead of 0.8). Adding `tracker="botsort.yaml"` to the call uses the tuned file.
- Timestamps come from `datetime.now().isoformat()`, which is local time with no UTC offset, and Supabase reads them as UTC. On a machine whose clock is not set to UTC the rows land hours off. West of UTC the two-minute summary window never sees them, so `bees_summary_per_minute` stays empty. More than two hours west (anywhere in the Americas), the cleanup also deletes them at its next hourly run. `datetime.now(timezone.utc).isoformat()` fixes it.
- `lap`, which the tracker needs, is not in `pyproject.toml`. Ultralytics installs it on the first run, and that needs internet access.
- The summary and cleanup threads print `Minute summary created` and `Deletion success` even when the call failed or Supabase is not configured.
- The docstring in `external_tracker.py` still refers to `live_tracker.py` and to `imgsz=640`. The script runs at 1024.

## Future improvements

- Add a web dashboard with historical playback and telemetry graphs, instead of just the HLS video.
- Upgrade the Raspberry Pi or move to cloud compute for future contained experiments.
- Advanced analytics: behavior classification, anomaly detection, etc.
- Optimize the model and tracking for better accuracy and speed.

## Tools used for training the model

https://github.com/jonathanrandall/yolo_labelling_tool
