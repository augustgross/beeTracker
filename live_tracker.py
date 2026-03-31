"""
Bee Tracker — Live + Supabase
==============================
Runs YOLO + BoT-SORT on a live camera feed and sends telemetry to Supabase.
Also saves data locally as backup.

Schema:
CREATE TABLE bees (
    bee_id INTEGER PRIMARY KEY
);

CREATE TABLE bee_frame_observation (
    observation_id BIGINT PRIMARY KEY,
    bee_id INTEGER NOT NULL REFERENCES bees(bee_id),
    frame INTEGER NOT NULL,
    time_stamp TIMESTAMPTZ NOT NULL,
    date DATE GENERATED ALWAYS AS ((time_stamp AT TIME ZONE 'UTC')::date) STORED,
    x_coord DOUBLE PRECISION,
    y_coord DOUBLE PRECISION,
    speed DOUBLE PRECISION
);

CREATE INDEX idx_bfo_timestamp 
ON bee_frame_observation(time_stamp);

CREATE INDEX idx_bfo_bee_timestamp 
ON bee_frame_observation(bee_id, time_stamp);

CREATE TABLE bees_summary_per_minute (
    bee_id INTEGER NOT NULL REFERENCES bees(bee_id),
    minute_bucket TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (bee_id, minute_bucket), 
    avg_speed DOUBLE PRECISION,
    mean_huddling DOUBLE PRECISION,
    avg_temp DOUBLE PRECISION
);

CREATE TABLE colony_summary_hourly (
    hour_bucket TIMESTAMPTZ PRIMARY KEY,
    avg_colony_speed DOUBLE PRECISION,
    total_huddle_clusters INTEGER,
    avg_cluster_size DOUBLE PRECISION
);



Usage:
    python live_tracker.py                  # use default camera (0)
    python live_tracker.py --camera 1       # use camera index 1
    python live_tracker.py --video test.mp4  # use a video file instead
"""

import cv2
import numpy as np
import os
import time
import argparse
import threading
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO
from supabase import create_client, Client

# Config

MODEL_PATH = "best.pt"
OUTPUT_DIR = Path("output")

FPS = 20.0
TIME_PER_FRAME = 1.0 / FPS

DISH_DIAMETER_METERS = 0.1
DISH_DIAMETER_PIXELS = 920.0
PIXELS_PER_METER = DISH_DIAMETER_PIXELS / DISH_DIAMETER_METERS

IDLE_SPEED_THRESHOLD = 0.01  # m/s

# How many rows to batch before uploading to Supabase to reduce API calls
UPLOAD_BATCH_SIZE = 300

COLORS = [(46, 204, 113), (52, 152, 219), (231, 76, 60),
           (241, 196, 15), (155, 89, 182)]


# Supabase Setup

load_dotenv()

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


# MINUTE BATCH
def update_minute_summary():
    """Aggregate recent telemetry into bees_summary_per_minute."""
    if db is None:
        return

    try:
        db.rpc("execute_sql", {
            "sql": """
            INSERT INTO bees_summary_per_minute (
                bee_id,
                minute_bucket,
                avg_speed,
                mean_huddling,
                avg_temp
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
        print(f"Supabase upload error: {e}")

#FOR DELETING OLD DATA (>HOUR OLD)
def flush_old_bee_frames():
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
        print(f"Deletion error: {e}")

#DELETION WORKER
def flush_worker():
    while True:
        flush_old_bee_frames()
        print(f"   Deletion success")
        time.sleep(3600)  # run every hour

#FOR MINUTE SUMMARY
def summary_worker():
    while True:
        update_minute_summary()
        print(f"  Minute summary created")
        time.sleep(60)

# Main Loop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--video", type=str, default=None, help="Video file instead of camera")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Open video source
    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERROR: Could not open video source: {source}")
        return

    if args.video:
        fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        print(f"Playing video: {args.video} ({fps} FPS)")
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
        fps = FPS
        print(f"Live camera {args.camera} at {fps} FPS")

    dt = 1.0 / fps

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Load model
    model = YOLO(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")

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

    print("Tracking started... (press Ctrl+C to stop)\n")

    #UPDATE MINUTE SUMMARY
    threading.Thread(target=summary_worker, daemon=True).start()
    #DELETE HOURLY
    threading.Thread(target=flush_worker, daemon=True).start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    break  # end of video file
                time.sleep(0.01)
                continue  # camera hiccup, try again

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

                # Write to local CSV
                csv_file.write(f"{now},{frame_num},{track_id},"
                               f"{center_x:.1f},{center_y:.1f},"
                               f"{speed_mps:.6f},{idle}\n")

                # Add to Supabase upload buffer
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

    # Flush remaining data
    if upload_buffer:
        upload_batch(upload_buffer)

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{frame_num} frames processed.")
    print(f"Local backup: {csv_path}")
    if db:
        print(f"Data uploaded to Supabase: bee_telemetry table")


if __name__ == "__main__":
    main()