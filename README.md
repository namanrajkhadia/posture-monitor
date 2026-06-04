# Posture Monitor

A tiny webcam-based posture coach for Windows. Calibrates against your good posture, then sends a quiet desktop notification when you slouch. Runs locally — no cloud, no account, no telemetry.

**[Download for Windows →](https://github.com/namanrajkhadia/posture-monitor/releases/latest)** (~130 MB zip · no Python required)

## Features

- **Calibrates to you.** A 5-second hold-still capture defines your good posture. Median-of-stable-samples baseline so brief flinches don't pollute it.
- **Quiet by design.** Lives in the system tray. Preview window can be hidden. The only interruption is one Windows toast when you actually slouch.
- **Per-axis sensitivity.** Five sliders for forward head, neck angle, head tilt, shoulder tilt, and alert hold time. Settings persist across runs.
- **Daily report.** "78% good posture today · 32 min good · 9 min bad" — one click in the tray menu.
- **Starts with Windows.** Optional, toggled from the tray menu.

## Install

### Easy: prebuilt zip

1. Download `PostureMonitor.zip` from the [latest release](https://github.com/namanrajkhadia/posture-monitor/releases/latest).
2. Right-click → Extract All.
3. Double-click `PostureMonitor.exe`.

### From source

```powershell
pip install mediapipe opencv-python winotify pystray Pillow pywin32

# pose model (~6 MB)
Invoke-WebRequest `
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task `
  -OutFile pose_landmarker_lite.task

# Desktop + Start Menu shortcut, no console
python install_shortcuts.py

# Launch
pythonw launch.pyw
```

## Files

- `posture_monitor.py` — main app: webcam loop, pose detection, evaluation, tray menu, persistence.
- `launch.pyw` — silent launcher that runs the app under `pythonw.exe` and logs crashes to `%APPDATA%\PostureMonitor\error.log`.
- `install_shortcuts.py` — creates Desktop and Start Menu shortcuts pointing at the launcher.
- `PostureMonitor.spec` — PyInstaller config that builds the standalone `.exe`.
- `index.html` — the project landing page.
- `headless_test.py` / `smoke_test.py` — sanity tests for the detection pipeline.

## Privacy

Webcam frames are read, fed to the MediaPipe pose detector in-process, and discarded. Nothing is recorded, saved, or transmitted. The only files written are:

- `%APPDATA%\PostureMonitor\config.json` — your sensitivity sliders and mute state.
- `%APPDATA%\PostureMonitor\stats.json` — per-day seconds in good vs. bad posture.
- `%APPDATA%\PostureMonitor\error.log` — only on crash.

## Build the .exe yourself

```powershell
pip install pyinstaller
python -m PyInstaller --noconfirm PostureMonitor.spec
Compress-Archive -Path "dist\PostureMonitor\*" -DestinationPath PostureMonitor.zip
```

Tested on Windows 11 with Python 3.13 and MediaPipe Tasks API.
