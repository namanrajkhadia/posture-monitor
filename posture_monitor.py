"""
Posture Monitor — webcam-based slouch detector for Windows.

Calibrates against your "good posture" on startup, then watches for:
  - forward head (ear moves forward of shoulder)
  - shoulder slouch (head drops toward shoulders, neck angle collapses)
  - excessive head/shoulder tilt

Triggers a Windows toast + system beep when bad posture persists past
the configured hold window, with a cooldown to avoid alert spam.

Runs as a tray app. The preview window has trackbars for per-axis sensitivity,
and the tray menu has Show/Hide preview, Recalibrate, Mute, Today's report,
and Start with Windows.

Hotkeys (with the preview window focused):
  q   quit
  r   recalibrate baseline
  m   toggle alerts (mute/unmute)
"""

import json
import math
import os
import sys
import threading
import time
import winreg
from collections import deque
from datetime import date

import cv2
import mediapipe as mp
import winsound
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)
from winotify import Notification

try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_TRAY = True
except ImportError:
    HAS_TRAY = False

# --- Paths & constants ---
CAM_INDEX = 0


def _bundle_dir():
    """Folder where bundled assets (model file, icon) live.

    - Source run: same dir as this file.
    - PyInstaller one-folder: same dir as the .exe.
    - PyInstaller one-file:    sys._MEIPASS extraction dir.
    """
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


APP_DIR = _bundle_dir()
MODEL_PATH = os.path.join(APP_DIR, "pose_landmarker_lite.task")

DATA_DIR = os.path.join(os.environ.get("APPDATA", APP_DIR), "PostureMonitor")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
STATS_PATH = os.path.join(DATA_DIR, "stats.json")

CALIBRATION_SECONDS = 5
CALIBRATION_READY_SECONDS = 2
ALERT_COOLDOWN_SECONDS = 30
SMOOTHING_WINDOW = 15

WINDOW_NAME = "Posture Monitor"
APP_NAME = "PostureMonitor"
RUN_KEY_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"

DEFAULT_CONFIG = {
    "forward_head_delta": 0.18,
    "neck_angle_delta_deg": 12.0,
    "head_tilt_delta_deg": 12.0,
    "shoulder_tilt_delta_deg": 8.0,
    "alert_hold_seconds": 5,
    "muted": False,
    "preview_visible": True,
}

FIX_TIPS = {
    "forward head":    "Pull chin back, ears over shoulders.",
    "neck compressed": "Lift the crown of your head; lengthen your neck.",
    "slouching":       "Stack ribcage over hips; roll shoulders back and down.",
    "head tilted":     "Level your head — eyes on the horizon.",
    "shoulders uneven":"Drop both shoulders down; even out your weight.",
}

# Landmark indices (MediaPipe Pose, 33-point)
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

POSE_EDGES = [
    (LEFT_EAR, RIGHT_EAR),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_EAR, LEFT_SHOULDER),
    (RIGHT_EAR, RIGHT_SHOULDER),
]
POSE_NODES = [NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER]


# --- Geometry helpers ---
def midpoint(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_deg(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


def extract_metrics(landmarks, w, h):
    """landmarks: list of NormalizedLandmark (33 entries). Returns dict or None."""
    needed = [LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER, NOSE]
    if any(landmarks[i].visibility < 0.5 for i in needed):
        return None

    def px(i):
        return (landmarks[i].x * w, landmarks[i].y * h)

    l_ear = px(LEFT_EAR)
    r_ear = px(RIGHT_EAR)
    l_sh = px(LEFT_SHOULDER)
    r_sh = px(RIGHT_SHOULDER)

    sh_mid = midpoint(l_sh, r_sh)
    ear_mid = midpoint(l_ear, r_ear)
    sh_width = max(dist(l_sh, r_sh), 1.0)

    forward_head_ratio = (ear_mid[0] - sh_mid[0]) / sh_width
    head_drop_ratio = (sh_mid[1] - ear_mid[1]) / sh_width

    dx = ear_mid[0] - sh_mid[0]
    dy = sh_mid[1] - ear_mid[1]
    neck_angle = math.degrees(math.atan2(dy, abs(dx) if dx != 0 else 1e-6))

    head_tilt = angle_deg(l_ear, r_ear)
    shoulder_tilt = angle_deg(l_sh, r_sh)

    return {
        "forward_head_ratio": forward_head_ratio,
        "head_drop_ratio": head_drop_ratio,
        "neck_angle": neck_angle,
        "head_tilt": head_tilt,
        "shoulder_tilt": shoulder_tilt,
    }


def average_metrics(samples):
    keys = samples[0].keys()
    return {k: sum(s[k] for s in samples) / len(samples) for k in keys}


def median_metrics(samples):
    """Per-axis median — robust to flinches/blinks during the calibration window."""
    keys = samples[0].keys()
    out = {}
    for k in keys:
        vals = sorted(s[k] for s in samples)
        n = len(vals)
        out[k] = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
    return out


def evaluate(metrics, baseline, cfg=None):
    """Returns (is_bad, reasons). cfg defaults to DEFAULT_CONFIG for backward compat."""
    if cfg is None:
        cfg = DEFAULT_CONFIG

    reasons = []

    fh_delta = metrics["forward_head_ratio"] - baseline["forward_head_ratio"]
    if abs(fh_delta) > cfg["forward_head_delta"]:
        reasons.append("forward head")

    head_drop_delta = metrics["head_drop_ratio"] - baseline["head_drop_ratio"]
    if head_drop_delta < -cfg["forward_head_delta"]:
        reasons.append("neck compressed")

    neck_delta = baseline["neck_angle"] - metrics["neck_angle"]
    if neck_delta > cfg["neck_angle_delta_deg"]:
        reasons.append("slouching")

    if abs(metrics["head_tilt"] - baseline["head_tilt"]) > cfg["head_tilt_delta_deg"]:
        reasons.append("head tilted")

    if abs(metrics["shoulder_tilt"] - baseline["shoulder_tilt"]) > cfg["shoulder_tilt_delta_deg"]:
        reasons.append("shoulders uneven")

    return (len(reasons) > 0, reasons)


def fix_tips_for(reasons):
    seen = []
    for r in reasons:
        tip = FIX_TIPS.get(r)
        if tip and tip not in seen:
            seen.append(tip)
    return seen


# --- Config / stats persistence ---
def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            stored = json.load(f)
        for k, v in stored.items():
            if k in cfg:
                cfg[k] = v
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return cfg


def save_config(cfg):
    _ensure_data_dir()
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG_PATH)


class StatsTracker:
    """Tracks per-day seconds in 'good' vs 'bad' posture, written to stats.json."""

    def __init__(self):
        self.data = self._load()
        self.last_tick = time.time()
        self.last_state = None
        self._dirty = False
        self._lock = threading.Lock()

    def _load(self):
        try:
            with open(STATS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def update(self, state):
        """state: 'good', 'bad', or 'idle'. Only good/bad accumulate."""
        now = time.time()
        elapsed = now - self.last_tick
        # Skip implausible gaps (sleep, paused) so they don't pollute the day.
        if self.last_state in ("good", "bad") and 0 < elapsed < 5.0:
            today = date.today().isoformat()
            with self._lock:
                bucket = self.data.setdefault(today, {"good": 0.0, "bad": 0.0})
                bucket[self.last_state] = bucket.get(self.last_state, 0.0) + elapsed
                self._dirty = True
        self.last_tick = now
        self.last_state = state

    def flush(self):
        with self._lock:
            if not self._dirty:
                return
            _ensure_data_dir()
            tmp = STATS_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
            os.replace(tmp, STATS_PATH)
            self._dirty = False

    def today_summary(self):
        today = date.today().isoformat()
        with self._lock:
            b = self.data.get(today, {"good": 0.0, "bad": 0.0})
            good = float(b.get("good", 0.0))
            bad = float(b.get("bad", 0.0))
        total = good + bad
        pct = (good / total * 100.0) if total > 0 else 0.0
        return good, bad, pct


# --- Autostart (HKCU Run key) ---
def autostart_command():
    pyexe = sys.executable
    # Prefer pythonw.exe so login start doesn't flash a console window.
    if pyexe.lower().endswith("python.exe"):
        candidate = pyexe[:-len("python.exe")] + "pythonw.exe"
        if os.path.exists(candidate):
            pyexe = candidate
    return f'"{pyexe}" "{os.path.abspath(__file__)}"'


def autostart_enabled():
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY_PATH) as k:
            winreg.QueryValueEx(k, APP_NAME)
        return True
    except OSError:
        return False


def set_autostart(enable):
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY_PATH, 0,
                        winreg.KEY_SET_VALUE) as k:
        if enable:
            winreg.SetValueEx(k, APP_NAME, 0, winreg.REG_SZ, autostart_command())
        else:
            try:
                winreg.DeleteValue(k, APP_NAME)
            except FileNotFoundError:
                pass


# --- Alerting ---
def send_alert(reasons):
    issues = ", ".join(reasons)
    tips = fix_tips_for(reasons)
    body = f"Detected: {issues}\n" + "\n".join(f"• {t}" for t in tips)
    try:
        winsound.Beep(880, 180)
        winsound.Beep(660, 180)
    except RuntimeError:
        pass
    try:
        Notification(
            app_id="Posture Monitor",
            title="Fix your posture",
            msg=body,
            duration="long",
        ).show()
    except Exception as e:
        print(f"[toast failed: {e}] {body}")


# --- Drawing ---
def draw_pose(frame, landmarks, w, h, color=(0, 255, 0)):
    pts = {}
    for i in POSE_NODES:
        lm = landmarks[i]
        if lm.visibility < 0.3:
            continue
        p = (int(lm.x * w), int(lm.y * h))
        pts[i] = p
        cv2.circle(frame, p, 4, color, -1)
    for a, b in POSE_EDGES:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], color, 2)


def detect(landmarker, frame_bgr, ts_ms):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(image, ts_ms)


# --- Calibration ---
def _stability_score(samples):
    """0..1 — higher means recent samples are stable enough to trust."""
    if len(samples) < 5:
        return 0.0
    recent = samples[-min(20, len(samples)):]
    keys_norms = {
        "forward_head_ratio": 0.05,
        "neck_angle": 5.0,
        "head_tilt": 5.0,
        "shoulder_tilt": 5.0,
    }
    parts = []
    for k, norm in keys_norms.items():
        vals = [s[k] for s in recent]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = var ** 0.5
        parts.append(max(0.0, 1.0 - std / norm))
    return sum(parts) / len(parts)


def _draw_quality_bar(frame, quality, sample_count, w, h):
    bar_w = int(w * 0.6)
    bar_x = (w - bar_w) // 2
    bar_y = h - 60
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20),
                  (60, 60, 60), -1)
    fill = int(bar_w * quality)
    if quality > 0.7:
        bar_color = (0, 220, 0)
    elif quality > 0.4:
        bar_color = (0, 200, 255)
    else:
        bar_color = (60, 100, 220)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + 20),
                  bar_color, -1)
    cv2.putText(frame,
                f"stability: {int(quality * 100)}%   samples: {sample_count}",
                (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (220, 220, 220), 1)


def calibrate(cap, landmarker, app, t0_ns):
    """Two-phase calibration: 'get ready' countdown, then capture phase."""
    print(f"Calibrating — get ready, then hold still for {CALIBRATION_SECONDS}s.")

    # Phase 1: "get ready" — show pose preview but don't capture yet
    ready_end = time.time() + CALIBRATION_READY_SECONDS
    while time.time() < ready_end:
        if app.stop_event.is_set():
            return None
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        ts_ms = int((time.time_ns() - t0_ns) / 1_000_000)
        res = detect(landmarker, frame, ts_ms)
        if res.pose_landmarks:
            draw_pose(frame, res.pose_landmarks[0], w, h, color=(0, 200, 255))

        remaining = max(0.0, ready_end - time.time())
        cv2.putText(frame, "Sit up straight — calibration starts in",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        big = f"{remaining:0.1f}"
        (tw, th), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 6)
        cv2.putText(frame, big, ((w - tw) // 2, (h + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 200, 255), 6)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None

    # Phase 2: capture
    samples = []
    end_time = time.time() + CALIBRATION_SECONDS
    while time.time() < end_time:
        if app.stop_event.is_set():
            return None
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        ts_ms = int((time.time_ns() - t0_ns) / 1_000_000)
        res = detect(landmarker, frame, ts_ms)

        pose_ok = False
        if res.pose_landmarks:
            lms = res.pose_landmarks[0]
            m = extract_metrics(lms, w, h)
            if m is not None:
                samples.append(m)
                pose_ok = True
            draw_pose(frame, lms, w, h,
                      color=(0, 220, 0) if pose_ok else (0, 200, 255))

        remaining = max(0.0, end_time - time.time())
        cv2.putText(frame, "HOLD STILL — capturing baseline",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
        big = f"{int(math.ceil(remaining))}"
        (tw, th), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 6)
        cv2.putText(frame, big, (w - tw - 30, (h + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 220, 255), 6)

        quality = _stability_score(samples)
        _draw_quality_bar(frame, quality, len(samples), w, h)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None

    if len(samples) < 10:
        print(f"Calibration failed — only {len(samples)} valid frames. "
              "Make sure your face and shoulders are in view.")
        return None

    # Median is robust to brief fidgets during the 5s capture (a small flinch
    # in 1 frame out of ~120 should not pull the baseline). Use the most-stable
    # 60% of frames so a worse-quality tail doesn't drag it either.
    quality_keys = ["forward_head_ratio", "neck_angle", "head_tilt"]
    rough = median_metrics(samples)
    scored = sorted(
        samples,
        key=lambda s: sum(abs(s[k] - rough[k]) for k in quality_keys),
    )
    keep = max(10, int(len(scored) * 0.6))
    baseline = median_metrics(scored[:keep])
    print(f"Baseline captured from {len(samples)} frames "
          f"(top-{keep} most stable used). "
          f"forward_head_ratio={baseline['forward_head_ratio']:+.3f}  "
          f"neck_angle={baseline['neck_angle']:.1f}°")
    return baseline


# --- Tray icon ---
def make_tray_image(state, muted):
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    color = {
        "good": (0, 200, 0, 255),
        "bad":  (220, 40, 40, 255),
        "idle": (140, 140, 140, 255),
    }.get(state, (140, 140, 140, 255))
    d.ellipse((6, 6, 58, 58), fill=color, outline=(20, 20, 20, 255), width=2)
    if muted:
        d.line((14, 14, 50, 50), fill=(0, 0, 0, 255), width=8)
        d.line((14, 14, 50, 50), fill=(255, 255, 255, 255), width=4)
    return img


def build_tray_menu(app):
    return pystray.Menu(
        pystray.MenuItem(
            "Show preview",
            lambda icon, item: app.set_preview(not item.checked),
            checked=lambda _: app.preview_visible,
            default=True,
        ),
        pystray.MenuItem("Recalibrate now",
                         lambda icon, item: app.request_recalibrate()),
        pystray.MenuItem("Reset thresholds to defaults",
                         lambda icon, item: app.reset_thresholds()),
        pystray.MenuItem(
            "Mute alerts",
            lambda icon, item: app.toggle_mute(),
            checked=lambda _: app.muted,
        ),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Today's posture report",
                         lambda icon, item: app.show_report()),
        pystray.MenuItem(
            "Start with Windows",
            lambda icon, item: app.toggle_autostart(),
            checked=lambda _: autostart_enabled(),
        ),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", lambda icon, item: app.shutdown()),
    )


# --- App state shared between cv2 main loop and tray thread ---
class PostureApp:
    def __init__(self):
        self.config = load_config()
        self.stats = StatsTracker()
        self.preview_visible = bool(self.config.get("preview_visible", True))
        self.muted = bool(self.config.get("muted", False))
        self.stop_event = threading.Event()
        self.recal_requested = False
        self.icon = None
        self.current_state = "idle"
        self._needs_window_rebuild = False

    def _persist(self):
        try:
            save_config(self.config)
        except OSError as e:
            print(f"[config save failed: {e}]")

    def set_preview(self, visible):
        self.preview_visible = bool(visible)
        self.config["preview_visible"] = self.preview_visible
        self._persist()
        if self.icon:
            self.icon.update_menu()

    def toggle_mute(self):
        self.muted = not self.muted
        self.config["muted"] = self.muted
        self._persist()
        if self.icon:
            self.icon.icon = make_tray_image(self.current_state, self.muted)
            self.icon.title = self._tray_title()
            self.icon.update_menu()

    def request_recalibrate(self):
        self.recal_requested = True
        self.preview_visible = True
        self.config["preview_visible"] = True
        self._persist()
        if self.icon:
            self.icon.update_menu()

    def show_report(self):
        good, bad, pct = self.stats.today_summary()
        total_min = (good + bad) / 60.0
        if total_min < 0.1:
            body = "No posture data yet today — get sitting!"
        else:
            body = (f"{pct:.0f}% good posture today\n"
                    f"{good/60:.1f} min good · {bad/60:.1f} min bad\n"
                    f"({total_min:.1f} min monitored)")
        try:
            Notification(app_id="Posture Monitor", title="Today's posture",
                         msg=body, duration="long").show()
        except Exception:
            print(body)

    def toggle_autostart(self):
        try:
            set_autostart(not autostart_enabled())
        except OSError as e:
            print(f"[autostart toggle failed: {e}]")
        if self.icon:
            self.icon.update_menu()

    def update_threshold(self, key, value):
        self.config[key] = float(value)
        self._persist()

    def reset_thresholds(self):
        threshold_keys = (
            "forward_head_delta",
            "neck_angle_delta_deg",
            "head_tilt_delta_deg",
            "shoulder_tilt_delta_deg",
            "alert_hold_seconds",
        )
        for k in threshold_keys:
            self.config[k] = DEFAULT_CONFIG[k]
        self._persist()
        # Force the preview window to be recreated so trackbars pick up new values.
        self.preview_visible = True
        self.config["preview_visible"] = True
        self._persist()
        self._needs_window_rebuild = True

    def update_state(self, state):
        if state == self.current_state:
            return
        self.current_state = state
        if self.icon:
            self.icon.icon = make_tray_image(state, self.muted)
            self.icon.title = self._tray_title()

    def _tray_title(self):
        label = {"good": "GOOD", "bad": "BAD", "idle": "no person"}.get(
            self.current_state, "")
        parts = [f"Posture: {label}"]
        if self.muted:
            parts.append("muted")
        return " · ".join(parts)

    def shutdown(self):
        self.stop_event.set()
        if self.icon:
            try:
                self.icon.stop()
            except Exception:
                pass


# --- Window + trackbars ---
def open_preview_window(app):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    # Sliders write back to config (and disk) on every drag.
    cv2.createTrackbar(
        "Fwd head x100", WINDOW_NAME,
        max(5, int(round(app.config["forward_head_delta"] * 100))), 50,
        lambda v: app.update_threshold("forward_head_delta", max(5, v) / 100.0))
    cv2.createTrackbar(
        "Neck deg", WINDOW_NAME,
        max(3, int(round(app.config["neck_angle_delta_deg"]))), 30,
        lambda v: app.update_threshold("neck_angle_delta_deg", max(3, v)))
    cv2.createTrackbar(
        "Head tilt deg", WINDOW_NAME,
        max(3, int(round(app.config["head_tilt_delta_deg"]))), 30,
        lambda v: app.update_threshold("head_tilt_delta_deg", max(3, v)))
    cv2.createTrackbar(
        "Shoulder deg", WINDOW_NAME,
        max(2, int(round(app.config["shoulder_tilt_delta_deg"]))), 25,
        lambda v: app.update_threshold("shoulder_tilt_delta_deg", max(2, v)))
    cv2.createTrackbar(
        "Hold sec", WINDOW_NAME,
        max(1, int(round(app.config["alert_hold_seconds"]))), 30,
        lambda v: app.update_threshold("alert_hold_seconds", max(1, v)))


def close_preview_window():
    try:
        cv2.destroyWindow(WINDOW_NAME)
    except cv2.error:
        pass


# --- Main loop ---
def main_loop(app, cap, landmarker, t0_ns):
    # Calibration always shows the preview so the user can see themselves.
    open_preview_window(app)
    window_open = True
    baseline = calibrate(cap, landmarker, app, t0_ns)
    if baseline is None:
        return
    if not app.preview_visible:
        close_preview_window()
        window_open = False

    smooth_buf = deque(maxlen=SMOOTHING_WINDOW)
    bad_since = None
    last_alert_time = 0.0
    last_stats_flush = time.time()

    print("Monitoring. Use the tray icon or 'q' / 'r' / 'm' in the preview.")

    while not app.stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        ts_ms = int((time.time_ns() - t0_ns) / 1_000_000)
        res = detect(landmarker, frame, ts_ms)

        status_text = "no person"
        status_color = (200, 200, 200)
        reasons = []
        state = "idle"

        if res.pose_landmarks:
            lms = res.pose_landmarks[0]
            m = extract_metrics(lms, w, h)
            if m is not None:
                smooth_buf.append(m)
                if len(smooth_buf) >= max(3, SMOOTHING_WINDOW // 2):
                    avg = average_metrics(list(smooth_buf))
                    is_bad, reasons = evaluate(avg, baseline, app.config)
                    now = time.time()
                    if is_bad:
                        state = "bad"
                        if bad_since is None:
                            bad_since = now
                        held = now - bad_since
                        status_text = f"BAD: {', '.join(reasons)}  ({held:0.1f}s)"
                        status_color = (0, 0, 255)
                        if (held >= app.config["alert_hold_seconds"]
                                and (now - last_alert_time) >= ALERT_COOLDOWN_SECONDS
                                and not app.muted):
                            send_alert(reasons)
                            last_alert_time = now
                    else:
                        state = "good"
                        bad_since = None
                        status_text = "GOOD"
                        status_color = (0, 200, 0)
            else:
                status_text = "low confidence"
            if window_open and app.preview_visible:
                draw_pose(frame, lms, w, h, color=status_color)

        app.update_state(state)
        app.stats.update(state)

        if time.time() - last_stats_flush > 10.0:
            app.stats.flush()
            last_stats_flush = time.time()

        # Reconcile preview window with the user's current preference.
        if app._needs_window_rebuild and window_open:
            close_preview_window()
            window_open = False
            app._needs_window_rebuild = False
        if app.preview_visible and not window_open:
            open_preview_window(app)
            window_open = True
        elif not app.preview_visible and window_open:
            close_preview_window()
            window_open = False

        if window_open:
            banner = f"{status_text}{'  [MUTED]' if app.muted else ''}"
            cv2.putText(frame, banner, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            if reasons:
                tips = fix_tips_for(reasons)
                for i, tip in enumerate(tips):
                    y = 80 + i * 28
                    cv2.putText(frame, f"-> {tip}", (20, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            good, bad, pct = app.stats.today_summary()
            if good + bad > 0:
                cv2.putText(frame, f"today: {pct:.0f}% good", (20, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 1)
            cv2.putText(frame, "q=quit  r=recalibrate  m=mute  (or use tray)",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (180, 180, 180), 1)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                app.shutdown()
                break
            elif key == ord('r'):
                app.recal_requested = True
            elif key == ord('m'):
                app.toggle_mute()
        else:
            time.sleep(0.05)

        if app.recal_requested:
            app.recal_requested = False
            if not window_open:
                open_preview_window(app)
                window_open = True
            smooth_buf.clear()
            bad_since = None
            new_baseline = calibrate(cap, landmarker, app, t0_ns)
            if new_baseline is not None:
                baseline = new_baseline


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}")
        print("Download it with: Invoke-WebRequest "
              "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task "
              f"-OutFile '{MODEL_PATH}'")
        return

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open webcam at index {CAM_INDEX}.")
        return

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    app = PostureApp()
    t0_ns = time.time_ns()

    tray_thread = None
    if HAS_TRAY:
        app.icon = pystray.Icon(
            APP_NAME,
            icon=make_tray_image("idle", app.muted),
            title="Posture Monitor",
            menu=build_tray_menu(app),
        )
        tray_thread = threading.Thread(target=app.icon.run, daemon=True)
        tray_thread.start()
    else:
        print("[pystray/Pillow not installed — running without tray icon]")

    try:
        main_loop(app, cap, landmarker, t0_ns)
    finally:
        app.stop_event.set()
        app.stats.flush()
        if app.icon:
            try:
                app.icon.stop()
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
