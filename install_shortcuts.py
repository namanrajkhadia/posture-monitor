"""
Create Desktop + Start Menu shortcuts that launch Posture Monitor silently.

Run once: python install_shortcuts.py
Run with --uninstall to remove them.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(HERE, "launch.pyw")
ICON_SRC = os.path.join(HERE, "icon.ico")
APP_NAME = "Posture Monitor"


def _pythonw_path():
    """Find pythonw.exe alongside the current interpreter."""
    pyexe = sys.executable
    if pyexe.lower().endswith("python.exe"):
        candidate = pyexe[:-len("python.exe")] + "pythonw.exe"
        if os.path.exists(candidate):
            return candidate
    if pyexe.lower().endswith("pythonw.exe"):
        return pyexe
    # Fallback: same dir as python.exe, named pythonw.exe
    cand = os.path.join(os.path.dirname(pyexe), "pythonw.exe")
    return cand if os.path.exists(cand) else pyexe


def _ensure_icon(force=False):
    """Generate icon.ico (gradient disc with stick-figure posture)."""
    if os.path.exists(ICON_SRC) and not force:
        return ICON_SRC
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    images = []
    for w, h in sizes:
        # Render at high res, downscale for smoothness on small sizes.
        scale = max(1, 256 // w)
        W, H = w * scale, h * scale
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        pad = max(1, W // 12)
        # Filled disc (background)
        d.ellipse((pad, pad, W - pad, H - pad),
                  fill=(15, 23, 42, 255),
                  outline=(125, 211, 252, 255),
                  width=max(2, W // 28))
        # Stick figure: head + shoulders + neck (good posture, ears over shoulders)
        cx = W // 2
        head_y = H // 3
        head_r = max(2, W // 11)
        sh_y = head_y + head_r * 2
        sh_half = W // 5
        accent = (125, 211, 252, 255)
        # Head
        d.ellipse((cx - head_r, head_y - head_r, cx + head_r, head_y + head_r),
                  fill=accent)
        # Neck
        line_w = max(2, W // 28)
        d.line((cx, head_y + head_r, cx, sh_y), fill=accent, width=line_w)
        # Shoulders
        d.line((cx - sh_half, sh_y, cx + sh_half, sh_y),
               fill=accent, width=line_w)
        # Spine (short)
        spine_bottom = sh_y + W // 5
        d.line((cx, sh_y, cx, spine_bottom), fill=accent, width=line_w)

        if scale > 1:
            img = img.resize((w, h), Image.LANCZOS)
        images.append(img)
    images[-1].save(ICON_SRC, format="ICO", sizes=sizes)
    return ICON_SRC


def _shortcut_paths():
    desktop = os.path.join(os.environ["USERPROFILE"], "Desktop")
    start_menu = os.path.join(
        os.environ["APPDATA"],
        "Microsoft", "Windows", "Start Menu", "Programs",
    )
    return [
        os.path.join(desktop, f"{APP_NAME}.lnk"),
        os.path.join(start_menu, f"{APP_NAME}.lnk"),
    ]


def _make_shortcut(lnk_path, target, args, icon, working_dir, description):
    """Create a .lnk via the WScript.Shell COM object — no extra deps."""
    import win32com.client  # pywin32; falls back below if missing

    shell = win32com.client.Dispatch("WScript.Shell")
    sc = shell.CreateShortcut(lnk_path)
    sc.TargetPath = target
    sc.Arguments = args
    sc.WorkingDirectory = working_dir
    sc.Description = description
    if icon:
        sc.IconLocation = icon
    sc.Save()


def _make_shortcut_powershell(lnk_path, target, args, icon, working_dir, description):
    """Fallback path: invoke PowerShell to create the .lnk so we don't need pywin32."""
    import subprocess

    ps = (
        f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{lnk_path}'); "
        f"$s.TargetPath = '{target}'; "
        f"$s.Arguments = '{args}'; "
        f"$s.WorkingDirectory = '{working_dir}'; "
        f"$s.Description = '{description}'; "
    )
    if icon:
        ps += f"$s.IconLocation = '{icon}'; "
    ps += "$s.Save()"
    subprocess.check_call(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
    )


def install():
    if not os.path.exists(LAUNCHER):
        print(f"launcher not found: {LAUNCHER}")
        sys.exit(1)

    pythonw = _pythonw_path()
    icon = _ensure_icon()
    args = f'"{LAUNCHER}"'

    try:
        import win32com.client  # noqa: F401
        make = _make_shortcut
    except ImportError:
        make = _make_shortcut_powershell

    for lnk in _shortcut_paths():
        os.makedirs(os.path.dirname(lnk), exist_ok=True)
        make(lnk, pythonw, args, icon, HERE, "Webcam-based posture monitor")
        print(f"created {lnk}")

    print(f"\nLaunch with: {pythonw} {args}")
    print("Or just double-click the new Desktop shortcut.")


def uninstall():
    for lnk in _shortcut_paths():
        if os.path.exists(lnk):
            os.remove(lnk)
            print(f"removed {lnk}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--uninstall", "-u"):
        uninstall()
    else:
        install()
