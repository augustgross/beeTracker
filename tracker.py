"""
Bee Tracking and Telemetry Extraction
=======================================
Uses YOLO11n for detection and BoT-SORT for consistent bee ID tracking.

Outputs (in output/ directory):
  1. bee_telemetry.csv            — per-frame positions and speed
  2. feeder_visits.csv            — per-visit feeder zone entry/exit log
  3. trophallaxis_events.csv      — detected trophallaxis interaction events
  4. bee_summary_statistics.xlsx  — per-bee speed, idle, feeder, and trophallaxis stats
  5. tracked_video.mp4            — annotated video with boxes, IDs, trails, and feeder zone
  6. speed_plot.png               — speed over time for all bees
"""

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

# ── Settings ──────────────────────────────────────────────────────────────────

VIDEO_PATH = "new_video.mp4"
MODEL_PATH = "new_model.pt"
OUTPUT_DIR = Path("output")

FPS = 20.0
TIME_PER_FRAME = 1.0 / FPS

DISH_DIAMETER_METERS = 0.1
DISH_DIAMETER_PIXELS = 920.0
PIXELS_PER_METER = DISH_DIAMETER_PIXELS / DISH_DIAMETER_METERS

IDLE_SPEED_THRESHOLD = 0.01  # m/s — below this a bee is considered stationary

# Feeder zone — rectangle (x1, y1, x2, y2) in pixel coordinates.
# Inspect the first frame of tracked_video.mp4 and adjust to surround the dark feeder blob.
FEEDER_ZONE = (170, 480, 470, 350)

# Trophallaxis detection thresholds
TROPHALLAXIS_DISTANCE_PIXELS = 120.0    # ~1.5× typical bee bounding-box width; tune as needed
TROPHALLAXIS_MIN_FRAMES = int(3 * FPS)  # minimum consecutive frames to confirm an event (3 s)

COLORS = [(46, 204, 113), (52, 152, 219), (231, 76, 60),
           (241, 196, 15), (155, 89, 182)]

OUTPUT_DIR.mkdir(exist_ok=True)

# ── YOLO tracking loop ────────────────────────────────────────────────────────

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

# Feeder visit state machine
# feeder_state: bee_id -> {in_zone, entry_frame, last_exit_frame}
feeder_state = {}
feeder_visits = []  # completed visit records

# Trophallaxis state machine
# pair_active: (id1, id2) -> {start_frame, frame_count, distances}
pair_active = {}
trophallaxis_events = []  # completed event records

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, verbose=False)
    vis_frame = frame.copy()

    # Pass 1 — compute speeds and collect all per-frame bee data before any
    # pair-wise analysis (trophallaxis requires the full frame snapshot).
    frame_bees = {}       # bee_id -> (cx, cy, speed_mps)
    frame_detections = [] # (x1, y1, x2, y2, track_id, cx, cy, speed_mps)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        track_id = int(box.id[0]) if box.id is not None else -1

        speed_mps = 0.0
        if track_id >= 0 and track_id in prev_positions:
            px, py = prev_positions[track_id]
            dist_pixels = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            speed_mps = (dist_pixels / PIXELS_PER_METER) / TIME_PER_FRAME

        if track_id >= 0:
            prev_positions[track_id] = (cx, cy)
            frame_bees[track_id] = (cx, cy, speed_mps)

        frame_detections.append((x1, y1, x2, y2, track_id, cx, cy, speed_mps))

    # Pass 2 — telemetry, feeder zone check, trails, drawing
    fz_x1, fz_y1, fz_x2, fz_y2 = FEEDER_ZONE

    for x1, y1, x2, y2, track_id, cx, cy, speed_mps in frame_detections:
        telemetry_rows.append({
            "frame": frame_num,
            "time_s": round(frame_num * TIME_PER_FRAME, 3),
            "bee_id": track_id,
            "x": round(cx, 1),
            "y": round(cy, 1),
            "speed_mps": round(speed_mps, 6)
        })

        if track_id >= 0:
            # Feeder zone state machine
            in_zone = fz_x1 <= cx <= fz_x2 and fz_y1 <= cy <= fz_y2

            if track_id not in feeder_state:
                feeder_state[track_id] = {
                    'in_zone': False,
                    'entry_frame': None,
                    'last_exit_frame': None
                }

            state = feeder_state[track_id]

            if in_zone and not state['in_zone']:
                # Bee just entered the feeder zone
                state['in_zone'] = True
                state['entry_frame'] = frame_num

            elif not in_zone and state['in_zone']:
                # Bee just exited the feeder zone
                entry_f = state['entry_frame']
                last_exit = state['last_exit_frame']
                feeder_visits.append({
                    'bee_id': track_id,
                    'entry_frame': entry_f,
                    'exit_frame': frame_num,
                    'dwell_frames': frame_num - entry_f,
                    'time_since_last_visit': (entry_f - last_exit) if last_exit is not None else None
                })
                state['in_zone'] = False
                state['last_exit_frame'] = frame_num
                state['entry_frame'] = None

            # Trails
            if track_id not in trails:
                trails[track_id] = []
            trails[track_id].append((cx, cy))
            if len(trails[track_id]) > 30:
                trails[track_id] = trails[track_id][-30:]

        color = COLORS[track_id % len(COLORS)] if track_id >= 0 else (0, 255, 0)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        label = f"Bee {track_id}" if track_id >= 0 else "?"
        cv2.putText(vis_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw trails
    for tid, trail in trails.items():
        color = COLORS[tid % len(COLORS)]
        for i in range(1, len(trail)):
            alpha = i / len(trail)
            thick = max(1, int(3 * alpha))
            pt1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
            pt2 = (int(trail[i][0]), int(trail[i][1]))
            cv2.line(vis_frame, pt1, pt2, color, thick)

    # Draw feeder zone overlay (orange rectangle) to aid calibration
    cv2.rectangle(vis_frame, (fz_x1, fz_y1), (fz_x2, fz_y2), (0, 165, 255), 2)
    cv2.putText(vis_frame, "Feeder", (fz_x1, fz_y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    # Pass 3 — trophallaxis pair analysis
    active_ids = list(frame_bees.keys())
    seen_pairs = set()

    for i in range(len(active_ids)):
        for j in range(i + 1, len(active_ids)):
            id1, id2 = active_ids[i], active_ids[j]
            pair = (min(id1, id2), max(id1, id2))
            seen_pairs.add(pair)

            cx1, cy1, spd1 = frame_bees[id1]
            cx2, cy2, spd2 = frame_bees[id2]
            dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

            conditions_met = (
                dist < TROPHALLAXIS_DISTANCE_PIXELS
                and spd1 < IDLE_SPEED_THRESHOLD
                and spd2 < IDLE_SPEED_THRESHOLD
            )

            if conditions_met:
                if pair not in pair_active:
                    pair_active[pair] = {
                        'start_frame': frame_num,
                        'frame_count': 0,
                        'distances': []
                    }
                pair_active[pair]['frame_count'] += 1
                pair_active[pair]['distances'].append(dist)
            else:
                if pair in pair_active:
                    rec = pair_active.pop(pair)
                    if rec['frame_count'] >= TROPHALLAXIS_MIN_FRAMES:
                        trophallaxis_events.append({
                            'bee_id_1': pair[0],
                            'bee_id_2': pair[1],
                            'start_frame': rec['start_frame'],
                            'end_frame': frame_num,
                            'duration_frames': rec['frame_count'],
                            'mean_distance': round(float(np.mean(rec['distances'])), 2)
                        })

    # Close trophallaxis events for pairs where a bee left the frame this tick
    for pair in list(pair_active.keys()):
        if pair not in seen_pairs:
            rec = pair_active.pop(pair)
            if rec['frame_count'] >= TROPHALLAXIS_MIN_FRAMES:
                trophallaxis_events.append({
                    'bee_id_1': pair[0],
                    'bee_id_2': pair[1],
                    'start_frame': rec['start_frame'],
                    'end_frame': frame_num,
                    'duration_frames': rec['frame_count'],
                    'mean_distance': round(float(np.mean(rec['distances'])), 2)
                })

    cv2.putText(vis_frame, f"Frame {frame_num}/{total_frames}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    writer.write(vis_frame)

    if frame_num % 100 == 0:
        n_det = len(results[0].boxes)
        print(f"  Frame {frame_num}/{total_frames}: {n_det} bees detected")

    frame_num += 1

cap.release()
writer.release()

last_frame = frame_num - 1  # index of last processed frame

# Close feeder visits still open at end of video
for bee_id, state in feeder_state.items():
    if state['in_zone'] and state['entry_frame'] is not None:
        entry_f = state['entry_frame']
        last_exit = state['last_exit_frame']
        feeder_visits.append({
            'bee_id': bee_id,
            'entry_frame': entry_f,
            'exit_frame': last_frame,
            'dwell_frames': last_frame - entry_f,
            'time_since_last_visit': (entry_f - last_exit) if last_exit is not None else None
        })

# Close trophallaxis events still active at end of video
for pair, rec in pair_active.items():
    if rec['frame_count'] >= TROPHALLAXIS_MIN_FRAMES:
        trophallaxis_events.append({
            'bee_id_1': pair[0],
            'bee_id_2': pair[1],
            'start_frame': rec['start_frame'],
            'end_frame': last_frame,
            'duration_frames': rec['frame_count'],
            'mean_distance': round(float(np.mean(rec['distances'])), 2)
        })

df = pd.DataFrame(telemetry_rows)
df.to_csv(OUTPUT_DIR / "bee_telemetry.csv", index=False)
print(f"  Done — {frame_num} frames processed")
print(f"  Saved: output/bee_telemetry.csv, output/tracked_video.mp4")

df_feeder = pd.DataFrame(feeder_visits)
if not df_feeder.empty:
    df_feeder.to_csv(OUTPUT_DIR / "feeder_visits.csv", index=False)
    print(f"  Saved: output/feeder_visits.csv ({len(df_feeder)} visits)")
else:
    print("  No feeder visits recorded (check FEEDER_ZONE coordinates).")

df_troph = pd.DataFrame(trophallaxis_events)
if not df_troph.empty:
    df_troph.to_csv(OUTPUT_DIR / "trophallaxis_events.csv", index=False)
    print(f"  Saved: output/trophallaxis_events.csv ({len(df_troph)} events)")
else:
    print("  No trophallaxis events recorded.")

# ── Summary statistics ────────────────────────────────────────────────────────

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

# Feeder visit stats per bee
if not df_feeder.empty:
    fv = df_feeder.groupby("bee_id").agg(
        _visits=("dwell_frames", "count"),
        _avg_dwell=("dwell_frames", "mean"),
        _avg_inter=("time_since_last_visit", "mean")
    )
    fv["Feeder Visits"] = fv["_visits"]
    fv["Avg Dwell (s)"] = (fv["_avg_dwell"] / FPS).round(2)
    fv["Avg Inter-Visit (s)"] = (fv["_avg_inter"] / FPS).round(2)
    bee_stats = bee_stats.join(
        fv[["Feeder Visits", "Avg Dwell (s)", "Avg Inter-Visit (s)"]],
        how="left"
    )
    bee_stats["Feeder Visits"] = bee_stats["Feeder Visits"].fillna(0).astype(int)

# Trophallaxis stats per bee
if not df_troph.empty:
    t_rows = []
    for bee_id in df_tracked["bee_id"].unique():
        involved = df_troph[
            (df_troph["bee_id_1"] == bee_id) | (df_troph["bee_id_2"] == bee_id)
        ]
        partners = (
            set(involved["bee_id_1"].tolist()) | set(involved["bee_id_2"].tolist())
        ) - {bee_id}
        t_rows.append({
            "bee_id": bee_id,
            "Trophallaxis Events": len(involved),
            "Unique Partners": len(partners),
            "Total Trophallaxis Time (s)": round(involved["duration_frames"].sum() / FPS, 2)
        })
    df_t_stats = pd.DataFrame(t_rows).set_index("bee_id")
    bee_stats = bee_stats.join(
        df_t_stats[["Trophallaxis Events", "Unique Partners", "Total Trophallaxis Time (s)"]],
        how="left"
    )
    bee_stats["Trophallaxis Events"] = bee_stats["Trophallaxis Events"].fillna(0).astype(int)
    bee_stats["Unique Partners"] = bee_stats["Unique Partners"].fillna(0).astype(int)

bee_stats.to_excel(OUTPUT_DIR / "bee_summary_statistics.xlsx")
print(f"  Saved: output/bee_summary_statistics.xlsx")
print(f"\n{bee_stats}\n")

# ── Speed plot ────────────────────────────────────────────────────────────────

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
