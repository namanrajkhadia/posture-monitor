"""Smoke test: open webcam, grab one frame, run pose, exit. No GUI."""
import sys
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("FAIL: webcam did not open"); sys.exit(1)

# Drain a few frames - some webcams need warm-up
ok, frame = False, None
for _ in range(10):
    ok, frame = cap.read()
    if ok and frame is not None:
        break
if not ok or frame is None:
    print("FAIL: could not read frame"); cap.release(); sys.exit(1)

h, w = frame.shape[:2]
print(f"webcam OK — frame {w}x{h}")

with mp.solutions.pose.Pose(model_complexity=1) as pose:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if res.pose_landmarks:
        n_visible = sum(1 for lm in res.pose_landmarks.landmark if lm.visibility > 0.5)
        print(f"pose OK — {n_visible}/33 landmarks visible")
    else:
        print("pose ran, but no landmarks detected (no person in view?)")

cap.release()
print("smoke test passed")
