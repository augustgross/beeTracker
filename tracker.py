"""
Bee Tracking and Telemetry Extraction
=======================================
Uses YOLO11n for detection and BoT-SORT for consistent bee ID tracking.

Outputs (in output/ directory):
  1. bee_telemetry.csv           — per-frame positions and speed
  2. bee_summary_statistics.xlsx  — per-bee avg/max speed, idle time
  3. tracked_video.mp4            — annotated video with boxes, IDs, and trails
  4. speed_plot.png               — speed over time for all bees
"""

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

# Settings

VIDEO_PATH = "2024-10-25_1848.mp4"
MODEL_PATH = "best.pt"
OUTPUT_DIR = Path("output")

FPS = 20.0
TIME_PER_FRAME = 1.0 / FPS

DISH_DIAMETER_METERS = 0.1
DISH_DIAMETER_PIXELS = 920.0
PIXELS_PER_METER = DISH_DIAMETER_PIXELS / DISH_DIAMETER_METERS

IDLE_SPEED_THRESHOLD = 0.01  # m/s

COLORS = [(46, 204, 113), (52, 152, 219), (231, 76, 60),
           (241, 196, 15), (155, 89, 182)]

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Run YOLO tracking and save telemetry

print("Running YOLO detection + BoT-SORT tracking...")

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = cv2.VideoWriter(str(OUTPUT_DIR / "tracked_video.mp4"),
                         cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))

telemetry_rows = []
trails = {}
prev_positions = {}

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, verbose=False)

    vis_frame = frame.copy()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        track_id = int(box.id[0]) if box.id is not None else -1

        speed_mps = 0.0
        if track_id >= 0 and track_id in prev_positions:
            px, py = prev_positions[track_id]
            dist_pixels = np.sqrt((center_x - px)**2 + (center_y - py)**2)
            dist_meters = dist_pixels / PIXELS_PER_METER
            speed_mps = dist_meters / TIME_PER_FRAME

        if track_id >= 0:
            prev_positions[track_id] = (center_x, center_y)

        telemetry_rows.append({
            "frame": frame_num,
            "time_s": round(frame_num * TIME_PER_FRAME, 3),
            "bee_id": track_id,
            "x": round(center_x, 1),
            "y": round(center_y, 1),
            "speed_mps": round(speed_mps, 6)
        })

        if track_id >= 0:
            if track_id not in trails:
                trails[track_id] = []
            trails[track_id].append((center_x, center_y))
            if len(trails[track_id]) > 30:
                trails[track_id] = trails[track_id][-30:]

        color = COLORS[track_id % len(COLORS)] if track_id >= 0 else (0, 255, 0)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        label = f"Bee {track_id}" if track_id >= 0 else "?"
        cv2.putText(vis_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for tid, trail in trails.items():
        color = COLORS[tid % len(COLORS)]
        for i in range(1, len(trail)):
            alpha = i / len(trail)
            thick = max(1, int(3 * alpha))
            pt1 = (int(trail[i-1][0]), int(trail[i-1][1]))
            pt2 = (int(trail[i][0]), int(trail[i][1]))
            cv2.line(vis_frame, pt1, pt2, color, thick)

    cv2.putText(vis_frame, f"Frame {frame_num}/{total_frames}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    writer.write(vis_frame)

    if frame_num % 100 == 0:
        n_det = len(results[0].boxes)
        print(f"  Frame {frame_num}/{total_frames}: {n_det} bees detected")

    frame_num += 1

cap.release()
writer.release()

df = pd.DataFrame(telemetry_rows)
df.to_csv(OUTPUT_DIR / "bee_telemetry.csv", index=False)
print(f"  Done — {frame_num} frames processed")
print(f"  Saved: output/bee_telemetry.csv, output/tracked_video.mp4")


# Summary statistics

print("\nGenerating summary statistics...")

df_tracked = df[df["bee_id"] >= 0].copy()

bee_stats = df_tracked.groupby("bee_id")["speed_mps"].agg(
    ["mean", "max", "count"]
).rename(columns={
    "mean": "Avg Speed (m/s)",
    "max": "Max Speed (m/s)",
    "count": "Frames Tracked"
})

idle_frames = df_tracked[df_tracked["speed_mps"] < IDLE_SPEED_THRESHOLD] \
    .groupby("bee_id").size()
bee_stats["Idle Frames"] = idle_frames.reindex(bee_stats.index, fill_value=0)
bee_stats["Idle Percentage (%)"] = (
    bee_stats["Idle Frames"] / bee_stats["Frames Tracked"] * 100
).round(2)

bee_stats.to_excel(OUTPUT_DIR / "bee_summary_statistics.xlsx")
print(f"  Saved: output/bee_summary_statistics.xlsx")
print(f"\n{bee_stats}\n")


# Speed plot

print("Generating speed plot...")

plt.figure(figsize=(12, 5))
for bee_id in sorted(df_tracked["bee_id"].unique()):
    bee_data = df_tracked[df_tracked["bee_id"] == bee_id]
    speed_mm = bee_data["speed_mps"] * 1000
    plt.plot(bee_data["time_s"], speed_mm, label=f"Bee {bee_id}", linewidth=1)

plt.xlabel("Time (s)")
plt.ylabel("Speed (mm/s)")
plt.title("Speed of all bees")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "speed_plot.png", dpi=150)
plt.close()
print(f"  Saved: output/speed_plot.png")

print("\nAll done!")