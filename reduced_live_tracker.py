"""
Bee Tracker — Raspberry Pi Deployment
=======================================
Runs YOLO11n (NCNN) + BoT-SORT on a live camera feed.
Sends telemetry to Supabase and saves video to external drive.

Before running:
  1. Export model:  python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='ncnn')"
  2. Mount drive:   sudo mount /dev/sda1 /mnt/storage
  3. Create .env with SUPABASE_URL and SUPABASE_KEY
  4. Run:           python live_tracker.py
  5. Or with video: python live_tracker.py --video test.mp4

Headless (no monitor):
  python live_tracker.py --no-display

As a systemd service (survives reboots):
  sudo cp bee-tracker.service /etc/systemd/system/
  sudo systemctl enable bee-tracker
  sudo systemctl start bee-tracker
"""

import cv2
import numpy as np
import os
import time
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO
from supabase import create_client, Client

# Config

MODEL_PATH = "model_ncnn_model"              # NCNN export for performance
OUTPUT_DIR = Path("/mnt/storage/bee_data")  # external drive
FALLBACK_DIR = Path("output")               # if external drive not mounted

FPS = 20.0
TIME_PER_FRAME = 1.0 / FPS

DISH_DIAMETER_METERS = 0.1
DISH_DIAMETER_PIXELS = 920.0
PIXELS_PER_METER = DISH_DIAMETER_PIXELS / DISH_DIAMETER_METERS

IDLE_SPEED_THRESHOLD = 0.01  # m/s

# Video recording
SAVE_VIDEO = False
VIDEO_CHUNK_HOURS = 1       # start new video file every hour
MAX_VIDEO_FILES = 500       # delete oldest when exceeded

# Supabase batching
UPLOAD_BATCH_SIZE = 100

# CSV backup, flush to disk periodically so data isn't lost on crash
CSV_FLUSH_INTERVAL = 50     # flush every N frames

COLORS = [(46, 204, 113), (52, 152, 219), (231, 76, 60),
           (241, 196, 15), (155, 89, 182)]


# Supabase Setup

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

db = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        db = create_client(SUPABASE_URL, SUPABASE_KEY)
        print(f"Connected to Supabase: {SUPABASE_URL}")
    except Exception as e:
        print(f"Supabase connection failed: {e}")
        print("Continuing with local-only storage.")
else:
    print("No Supabase credentials - saving locally only.")


def upload_batch(rows):
    """Upload a batch of telemetry rows to Supabase."""
    if db is None or not rows:
        return
    try:
        db.table("bee_frame_observation").insert(rows).execute()
    except Exception as e:
        print(f"  Supabase upload error: {e}")


# Storage Management

def cleanup_old_videos(video_dir):
    """Delete oldest video files if exceed MAX_VIDEO_FILES."""
    videos = sorted(video_dir.glob("recording_*.mp4"), key=lambda f: f.stat().st_mtime)
    while len(videos) > MAX_VIDEO_FILES:
        oldest = videos.pop(0)
        oldest.unlink()
        print(f"  Deleted old video: {oldest.name}")


# Main

def main():
    parser = argparse.ArgumentParser(description="Bee Tracker - Pi Deployment")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--video", type=str, default=None, help="Video file instead of camera")
    parser.add_argument("--no-video", action="store_true", help="Skip saving video")
    parser.add_argument("--no-display", action="store_true", help="No GUI window (headless)")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Model path")
    args = parser.parse_args()

    save_video = SAVE_VIDEO and not args.no_video

    # Choose output directory
    out_dir = OUTPUT_DIR
    if not OUTPUT_DIR.parent.exists():
        print(f"External drive not found at {OUTPUT_DIR.parent}, using {FALLBACK_DIR}")
        out_dir = FALLBACK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open video source
    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERROR: Could not open video source: {source}")
        return

    if args.video:
        fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Playing video: {args.video} ({total} frames, {fps} FPS)")
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
        fps = FPS
        print(f"Live camera {args.camera}")

    dt = 1.0 / fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution: {w}x{h}")

    # Load model
    model = YOLO(args.model)
    print(f"Model loaded: {args.model}")

    # Local CSV backup
    csv_path = out_dir / f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file = open(csv_path, "w")
    csv_file.write("timestamp,frame,bee_id,x,y,speed_mps,idle\n")

    # Video writer with hourly chunks
    writer = None
    chunk_start = None

    def start_new_chunk():
        nonlocal writer, chunk_start
        if writer:
            writer.release()
        chunk_name = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        writer = cv2.VideoWriter(
            str(out_dir / chunk_name),
            cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        chunk_start = time.time()
        cleanup_old_videos(out_dir)
        print(f"  New video chunk: {chunk_name}")

    if save_video:
        start_new_chunk()

    # Tracking state
    prev_positions = {}
    trails = {}
    upload_buffer = []
    frame_num = 0
    start_time = time.time()

    print("Tracking started... (press Ctrl+C to stop)\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    break
                time.sleep(0.01)
                continue

            # Run YOLO + BoT-SORT
            results = model.track(frame, persist=True, verbose=False)

            vis_frame = frame.copy()
            now = datetime.now().isoformat()

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                track_id = int(box.id[0]) if box.id is not None else -1

                # Speed
                speed_mps = 0.0
                if track_id >= 0 and track_id in prev_positions:
                    px, py = prev_positions[track_id]
                    dist_px = np.sqrt((center_x - px)**2 + (center_y - py)**2)
                    speed_mps = (dist_px / PIXELS_PER_METER) / dt

                if track_id >= 0:
                    prev_positions[track_id] = (center_x, center_y)

                idle = speed_mps < IDLE_SPEED_THRESHOLD

                # Local CSV
                csv_file.write(f"{now},{frame_num},{track_id},"
                               f"{center_x:.1f},{center_y:.1f},"
                               f"{speed_mps:.6f},{idle}\n")

                # Supabase buffer (only tracked bees)
                if track_id >= 0:
                    upload_buffer.append({
                        "bee_id": track_id,
                        "frame": frame_num,
                        "time_stamp": now,
                        "x_coord": round(center_x, 1),
                        "y_coord": round(center_y, 1),
                        "speed": round(speed_mps, 6),
                    })

                # Trails
                if track_id >= 0:
                    if track_id not in trails:
                        trails[track_id] = []
                    trails[track_id].append((center_x, center_y))
                    if len(trails[track_id]) > 30:
                        trails[track_id] = trails[track_id][-30:]

                # Draw
                color = COLORS[track_id % len(COLORS)] if track_id >= 0 else (0, 255, 0)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis_frame, f"Bee {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw trails
            for tid, trail in trails.items():
                color = COLORS[tid % len(COLORS)]
                for i in range(1, len(trail)):
                    thick = max(1, int(3 * i / len(trail)))
                    pt1 = (int(trail[i-1][0]), int(trail[i-1][1]))
                    pt2 = (int(trail[i][0]), int(trail[i][1]))
                    cv2.line(vis_frame, pt1, pt2, color, thick)

            # HUD
            elapsed = time.time() - start_time
            actual_fps = frame_num / elapsed if elapsed > 0 else 0
            cv2.putText(vis_frame, f"Frame {frame_num} | {actual_fps:.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Save video frame + rotate hourly
            if save_video and writer:
                writer.write(vis_frame)
                if time.time() - chunk_start > VIDEO_CHUNK_HOURS * 3600:
                    start_new_chunk()

            # Display (skip if headless)
            if not args.no_display:
                cv2.imshow("Bee Tracker", vis_frame)
                if cv2.waitKey(1) == 27:
                    break

            # Upload batch to Supabase
            if len(upload_buffer) >= UPLOAD_BATCH_SIZE:
                upload_batch(upload_buffer)
                upload_buffer = []

            # Flush CSV to disk
            if frame_num % CSV_FLUSH_INTERVAL == 0:
                csv_file.flush()

            # Progress log
            if frame_num % 500 == 0 and frame_num > 0:
                disk_kb = csv_path.stat().st_size // 1024
                print(f"  Frame {frame_num} | {actual_fps:.1f} FPS | "
                      f"{len(prev_positions)} bees | CSV: {disk_kb} KB")

            frame_num += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nError: {e}")

    # Cleanup
    if upload_buffer:
        upload_batch(upload_buffer)

    csv_file.close()
    if writer:
        writer.release()
    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"\n{frame_num} frames in {elapsed:.0f}s ({frame_num/elapsed:.1f} FPS)")
    print(f"Local backup: {csv_path}")
    if db:
        print("Data uploaded to Supabase: bee_frame_observation table")


if __name__ == "__main__":
    main()