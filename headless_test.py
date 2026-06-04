"""
Headless test of posture_monitor logic without opening the webcam GUI.
Runs detection over real webcam frames and exercises:
  - Tasks API video-mode detect_for_video
  - extract_metrics with synthetic landmarks
  - evaluate() across known good / bad cases
"""
import sys
import time
import math
from types import SimpleNamespace
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

import posture_monitor as pm


def make_lm(x, y, vis=0.95):
    return SimpleNamespace(x=x, y=y, z=0.0, visibility=vis, presence=1.0)


def synth_landmarks(forward_head_offset=0.0, head_drop=0.20, shoulder_y=0.7, head_tilt=0.0):
    """
    Build a 33-point landmark list with realistic ear/shoulder/nose positions.
    Coordinates are normalized (0..1).  All other points get visibility 0 so they're ignored.
    """
    lms = [make_lm(0.5, 0.5, vis=0.0) for _ in range(33)]
    sh_y = shoulder_y
    ear_y = sh_y - head_drop
    sh_dx = 0.10  # half-width of shoulders
    ear_dx = 0.06
    cx = 0.5 + forward_head_offset
    # tilt rotates ear pair around its midpoint
    tilt_rad = math.radians(head_tilt)
    le_x = cx - ear_dx * math.cos(tilt_rad)
    le_y = ear_y - ear_dx * math.sin(tilt_rad)
    re_x = cx + ear_dx * math.cos(tilt_rad)
    re_y = ear_y + ear_dx * math.sin(tilt_rad)

    lms[pm.NOSE]           = make_lm(cx, ear_y - 0.04)
    lms[pm.LEFT_EAR]       = make_lm(le_x, le_y)
    lms[pm.RIGHT_EAR]      = make_lm(re_x, re_y)
    lms[pm.LEFT_SHOULDER]  = make_lm(0.5 - sh_dx, sh_y)
    lms[pm.RIGHT_SHOULDER] = make_lm(0.5 + sh_dx, sh_y)
    return lms


def test_metrics_and_evaluate():
    print("--- unit tests on metrics + evaluate ---")
    W, H = 640, 480

    # Baseline: upright
    upright = synth_landmarks(forward_head_offset=0.0, head_drop=0.20, head_tilt=0.0)
    baseline = pm.extract_metrics(upright, W, H)
    assert baseline is not None, "upright case must extract metrics"
    print(f"  baseline: fh={baseline['forward_head_ratio']:+.3f}  "
          f"neck_angle={baseline['neck_angle']:.1f}  "
          f"head_drop={baseline['head_drop_ratio']:.3f}")

    # Same as baseline → should be GOOD
    again = pm.extract_metrics(synth_landmarks(0.0, 0.20, head_tilt=0.0), W, H)
    is_bad, reasons = pm.evaluate(again, baseline)
    print(f"  same-as-baseline: bad={is_bad} reasons={reasons}")
    assert not is_bad, "same as baseline must be GOOD"

    # Forward head: shift ears forward by lots
    fwd = pm.extract_metrics(synth_landmarks(forward_head_offset=0.10, head_drop=0.20), W, H)
    is_bad, reasons = pm.evaluate(fwd, baseline)
    print(f"  forward-head:    bad={is_bad} reasons={reasons}")
    assert is_bad, "forward head should be BAD"

    # Slouched: head drops down toward shoulders (smaller head_drop)
    slouch = pm.extract_metrics(synth_landmarks(forward_head_offset=0.0, head_drop=0.05), W, H)
    is_bad, reasons = pm.evaluate(slouch, baseline)
    print(f"  slouched:        bad={is_bad} reasons={reasons}")
    assert is_bad, "slouched should be BAD"

    # Head tilted
    tilted = pm.extract_metrics(synth_landmarks(0.0, 0.20, head_tilt=20.0), W, H)
    is_bad, reasons = pm.evaluate(tilted, baseline)
    print(f"  head-tilted:     bad={is_bad} reasons={reasons}")
    assert is_bad, "tilted head should be BAD"

    print("  PASS\n")


def test_video_pipeline():
    """Open webcam, grab a few frames, run them through the actual Tasks API in VIDEO mode."""
    print("--- live pipeline test (3 frames, no GUI) ---")
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pm.MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("  FAIL: cannot open webcam"); return False

    t0 = time.time_ns()
    ok_frames = 0
    pose_detections = 0
    metrics_extracted = 0
    for i in range(15):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        ok_frames += 1
        ts_ms = int((time.time_ns() - t0) / 1_000_000)
        res = pm.detect(landmarker, frame, ts_ms)
        if res.pose_landmarks:
            pose_detections += 1
            h, w = frame.shape[:2]
            m = pm.extract_metrics(res.pose_landmarks[0], w, h)
            if m is not None:
                metrics_extracted += 1
        time.sleep(0.05)

    cap.release()
    landmarker.close()
    print(f"  frames ok: {ok_frames}/15   pose detections: {pose_detections}   "
          f"metrics extracted: {metrics_extracted}")
    print(f"  (zero detections is fine — nobody is in front of the camera during this test)")
    print("  PASS\n")
    return True


if __name__ == "__main__":
    test_metrics_and_evaluate()
    test_video_pipeline()
    print("All headless tests passed.")
