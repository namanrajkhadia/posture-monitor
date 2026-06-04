"""
Silent launcher for Posture Monitor.

Double-clicking this file (or a shortcut to it) runs the app via pythonw.exe,
which means no console window. Any unhandled crash is written to
%APPDATA%\\PostureMonitor\\error.log so silent failures are still debuggable.
"""
import os
import sys
import traceback
from datetime import datetime

# Ensure imports resolve even when launched from a shortcut with a different cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _log_crash(exc):
    log_dir = os.path.join(os.environ.get("APPDATA", HERE), "PostureMonitor")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "error.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n--- {datetime.now().isoformat()} ---\n")
        f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    # Best-effort toast so the user actually sees something happened.
    try:
        from winotify import Notification
        Notification(
            app_id="Posture Monitor",
            title="Posture Monitor crashed",
            msg=f"See {log_path}",
            duration="long",
        ).show()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        import posture_monitor
        posture_monitor.main()
    except Exception as e:
        _log_crash(e)
        raise
