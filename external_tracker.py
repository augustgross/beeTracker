"""
Bee Tracker — Live RTSP + Supabase
====================================
Pulls frames from a Pi-hosted RTSP stream, runs YOLO + BoT-SORT, and pushes
telemetry to Supabase. Local CSV is written as backup.

Key fixes vs. previous version:
  * LatestFrameGrabber drops stale buffered frames so we always process
    the freshest frame the stream has produced (instead of falling behind).
  * Wall-clock dt is used for speed calculation, since frames are now skipped
    when inference is slower than the stream's frame rate.
  * imgsz=640 on inference for ~2× speedup vs. default 1024.
  * OPENCV_FFMPEG_CAPTURE_OPTIONS is set BEFORE importing cv2 (it has no
    effect if set after).

Usage:
    python live_tracker.py                          # default RTSP URL
    python live_tracker.py --rtsp rtsp://IP:8554/cam
    python live_tracker.py --video test.mp4         # local file instead
"""

import os

# MUST be set before importing cv2 — tells OpenCV's FFmpeg backend to use
# TCP transport and skip its default 5-frame input buffer.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|"
    "framedrop;1|max_delay;0"
)

import cv2
import numpy as np
import time
import argparse
import threading
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO
from supabase import create_client, Client
import torch



# Config
load_dotenv()
RTSP_URL=os.getenv("RTSP_URL")
MODEL_PATH = "bee.pt"
INFERENCE_IMGSZ = 1024
OUTPUT_DIR = Path("output")

FPS = 20.0                   # nominal fallback if source doesn't report

DISH_DIAMETER_METERS = 0.1
DISH_DIAMETER_PIXELS = 920.0
PIXELS_PER_METER = DISH_DIAMETER_PIXELS / DISH_DIAMETER_METERS

IDLE_SPEED_THRESHOLD = 0.01  # m/s

UPLOAD_BATCH_SIZE = 300

COLORS = [(46, 204, 113), (52, 152, 219), (231, 76, 60),
          (241, 196, 15), (155, 89, 182)]


# RTSP Reader

class LatestFrameGrabber:
    """
    Background thread that always holds the most recent frame.
    Older frames are dropped, not queued. This is critical when inference
    is slower than the stream's frame rate, otherwise OpenCV hands you
    increasingly stale frames as its internal buffer fills.

    Mimics enough of cv2.VideoCapture's interface that the main loop
    (cap.read, cap.release, cap.isOpened) doesn't need to change.
    """
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open RTSP stream: {url}")
        self.latest = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            with self.lock:
                self.latest = frame  # overwrite — never queue

    def read(self):
        with self.lock:
            if self.latest is None:
                return False, None
            return True, self.latest.copy()

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


# Supabase Setup



SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("WARNING: SUPABASE_URL and SUPABASE_KEY not set in .env")
    print("         Data will only be saved locally.")
    db = None
else:
    db: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(f"Connected to Supabase: {SUPABASE_URL}")


def upload_batch(rows):
    """Upload a batch of telemetry rows to Supabase."""
    if db is None or not rows:
        return
    try:
        db.table("bee_frame_observation").insert(rows).execute()
    except Exception as e:
        print(f"  Supabase upload error: {e}")


def update_minute_summary():
    """Aggregate recent telemetry into bees_summary_per_minute."""
    if db is None:
        return
    try:
        db.rpc("execute_sql", {
            "sql": """
            INSERT INTO bees_summary_per_minute (
                bee_id, minute_bucket, avg_speed, mean_huddling, avg_temp
            )
            SELECT
                bee_id,
                date_trunc('minute', time_stamp) AS minute_bucket,
                AVG(speed) AS avg_speed,
                NULL::double precision AS mean_huddling,
                NULL::double precision AS avg_temp
            FROM bee_frame_observation
            WHERE time_stamp >= NOW() - INTERVAL '2 minutes'
            GROUP BY bee_id, date_trunc('minute', time_stamp)
            ON CONFLICT (bee_id, minute_bucket)
            DO UPDATE SET
                avg_speed = EXCLUDED.avg_speed,
                mean_huddling = EXCLUDED.mean_huddling,
                avg_temp = EXCLUDED.avg_temp;
            """
        }).execute()
    except Exception as e:
        print(f"  Minute summary error: {e}")


def flush_old_bee_frames():
    """Delete bee_frame_observation rows older than 2 hours."""
    if db is None:
        return
    try:
        db.rpc("execute_sql", {
            "sql": """
            DELETE FROM bee_frame_observation
            WHERE time_stamp < NOW() - INTERVAL '2 hours';
            """
        }).execute()
    except Exception as e:
        print(f"  Deletion error: {e}")


def flush_worker():
    while True:
        flush_old_bee_frames()
        print("  Deletion success")
        time.sleep(3600)


def summary_worker():
    while True:
        update_minute_summary()
        print("  Minute summary created")
        time.sleep(60)


# Main loop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp", type=str, default=RTSP_URL, help="RTSP URL")
    parser.add_argument("--video", type=str, default=None,
                        help="Local video file instead of RTSP")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless mode (no GUI window)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Open source
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"ERROR: Could not open video file: {args.video}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        print(f"Playing video: {args.video} ({fps:.1f} FPS)")
        is_live = False
    else:
        try:
            cap = LatestFrameGrabber(args.rtsp)
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return
        fps = FPS  # nominal — actual rate is governed by inference speed
        print(f"RTSP stream: {args.rtsp}")
        is_live = True

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH)
    model.to(device)
    print(f"Model loaded on {device}: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    # print(f"Model loaded: {MODEL_PATH} (imgsz={INFERENCE_IMGSZ})")

    # Local CSV backup
    csv_path = OUTPUT_DIR / f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file = open(csv_path, "w")
    csv_file.write("timestamp,frame,bee_id,x,y,speed_mps,idle\n")

    # State
    prev_positions = {}
    trails = {}
    upload_buffer = []
    frame_num = 0
    start_time = time.time()
    last_frame_time = time.time()

    # Background workers (Supabase aggregation + cleanup)
    threading.Thread(target=summary_worker, daemon=True).start()
    threading.Thread(target=flush_worker, daemon=True).start()

    print("Tracking started... (press ESC or Ctrl+C to stop)\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if not is_live:
                    break  # end of video file
                time.sleep(0.01)
                continue  # stream hiccup, try again

            # Wall-clock dt — frames may be skipped by LatestFrameGrabber,
            # so 1/fps is no longer valid for speed calculations.
            now_t = time.time()
            dt = now_t - last_frame_time
            last_frame_time = now_t
            if dt <= 0 or dt > 1.0:
                dt = 1.0 / fps  # guard against startup or long stalls

            # Inference (smaller imgsz = faster)
            # results = model.track(frame, persist=True, verbose=False,
            #                       imgsz=INFERENCE_IMGSZ)
            results = model.track(frame, persist=True, verbose=False,
                      imgsz=INFERENCE_IMGSZ, device=device, half=True)

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

                # Local CSV — write everything (including untracked detections)
                csv_file.write(f"{now},{frame_num},{track_id},"
                               f"{center_x:.1f},{center_y:.1f},"
                               f"{speed_mps:.6f},{idle}\n")

                # Supabase — only tracked bees (untracked track_id=-1 would
                # violate the bees(bee_id) foreign key)
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

                # Draw box + label
                color = COLORS[track_id % len(COLORS)] if track_id >= 0 else (0, 255, 0)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis_frame, f"Bee {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Trails
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
            cv2.putText(vis_frame, f"Frame {frame_num} | {actual_fps:.1f} FPS",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if not args.no_display:
                cv2.imshow("Bee Tracker", vis_frame)
                if cv2.waitKey(1) == 27:
                    break

            # Upload batch to Supabase
            if len(upload_buffer) >= UPLOAD_BATCH_SIZE:
                upload_batch(upload_buffer)
                upload_buffer = []

            if frame_num % 500 == 0 and frame_num > 0:
                print(f"  Frame {frame_num} | {actual_fps:.1f} FPS | "
                      f"{len(prev_positions)} bees tracked")

            frame_num += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        import traceback
        traceback.print_exc()

    # Cleanup
    if upload_buffer:
        upload_batch(upload_buffer)

    csv_file.close()
    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    final_fps = frame_num / elapsed if elapsed > 0 else 0
    print(f"\n{frame_num} frames in {elapsed:.1f}s ({final_fps:.1f} FPS)")
    print(f"Local backup: {csv_path}")
    if db:
        print("Data uploaded to Supabase: bee_frame_observation table")


if __name__ == "__main__":
    main()