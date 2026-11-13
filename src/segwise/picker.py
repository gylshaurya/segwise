"""Native file and folder pickers for the local web app.

A browser page cannot see real file paths, and segwise reads files where they
are (no uploads), so the app asks the operating system to show its own picker:

* macOS: the Finder dialog via AppleScript (osascript)
* Windows: the standard Windows dialog via Tk, in a separate process
* Linux: zenity (GNOME) or kdialog (KDE), falling back to Tk

Each dialog runs in its own process, so it never blocks or crashes the server.
Returns the chosen path, or None when the person cancels.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

KINDS = ("file", "folder", "save", "nav")
SEGY_PATTERNS = "*.sgy *.segy *.seg *.su *.SGY *.SEGY *.SEG *.SU"
NAV_PATTERNS = "*.p190 *.P190 *.190 *.nav *.NAV *.txt *.csv *.dat *.410"
TITLES = {"file": "Choose a SEG-Y file", "folder": "Choose a folder of SEG-Y files",
          "save": "Save the fixed copy as", "nav": "Choose the navigation file"}


class PickerUnavailable(RuntimeError):
    pass


def _start_dir(start: str | None) -> str:
    if start:
        s = os.path.expanduser(start)
        if os.path.isfile(s):
            s = os.path.dirname(s)
        if os.path.isdir(s):
            return s
    return os.path.expanduser("~")


def _as(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _mac(kind: str, start: str | None) -> str | None:
    loc = f' default location (POSIX file "{_as(_start_dir(start))}")'
    prompt = f'with prompt "{_as(TITLES[kind])}"'
    if kind == "folder":
        expr = f"choose folder {prompt}{loc}"
    elif kind == "save":
        name = os.path.basename(start) if start and not os.path.isdir(os.path.expanduser(start)) else "fixed_std.sgy"
        expr = f'choose file name {prompt} default name "{_as(name)}"{loc}'
    else:
        expr = f"choose file {prompt}{loc}"
    # "tell me to activate" brings the dialog in front of the browser without extra permissions
    r = subprocess.run(["osascript", "-e", "tell me to activate", "-e", f"POSIX path of ({expr})"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        if "-128" in r.stderr:  # User canceled
            return None
        raise PickerUnavailable(r.stderr.strip() or "osascript failed")
    return r.stdout.strip() or None


def _linux(kind: str, start: str | None) -> str | None:
    d = _start_dir(start) + "/"
    if shutil.which("zenity"):
        cmd = ["zenity", "--file-selection", f"--title={TITLES[kind]}", f"--filename={d}"]
        if kind == "folder":
            cmd.append("--directory")
        elif kind == "save":
            cmd += ["--save", "--confirm-overwrite"]
        else:
            cmd += [f"--file-filter=SEG-Y | {SEGY_PATTERNS}" if kind == "file" else f"--file-filter=Navigation | {NAV_PATTERNS}",
                    "--file-filter=All files | *"]
    elif shutil.which("kdialog"):
        flag = {"folder": "--getexistingdirectory", "save": "--getsavefilename"}.get(kind, "--getopenfilename")
        cmd = ["kdialog", "--title", TITLES[kind], flag, d]
        if kind == "file":
            cmd.append(f"{SEGY_PATTERNS}|SEG-Y files")
        elif kind == "nav":
            cmd.append(f"{NAV_PATTERNS}|Navigation files")
    else:
        return _tk_subprocess(kind, start)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 1:
        return None  # cancelled
    if r.returncode != 0:
        raise PickerUnavailable(r.stderr.strip() or f"{cmd[0]} failed")
    return r.stdout.strip() or None


def _tk_subprocess(kind: str, start: str | None) -> str | None:
    """Run the Tk dialog in a child process (Tk must own the main thread of its process)."""
    if getattr(sys, "frozen", False):  # the packaged app re-launches itself with a hidden command
        cmd = [sys.executable, "_pick", kind, start or ""]
    else:
        cmd = [sys.executable, "-m", "segwise", "_pick", kind, start or ""]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    r = subprocess.run(cmd, capture_output=True, text=True, creationflags=flags)
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        raise PickerUnavailable((r.stderr or "the file dialog could not start").strip().splitlines()[-1])
    if out.get("error"):
        raise PickerUnavailable(out["error"])
    return out.get("path") or None


def tk_dialog(kind: str, start: str | None) -> None:
    """Entry point of the child process: show the dialog and print {"path": ...} as JSON."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:  # pragma: no cover - Tk missing (some Linux installs)
        print(json.dumps({"error": f"No file dialog available ({e}). Paste the path instead."}))
        return
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    d = _start_dir(start)
    if kind == "folder":
        p = filedialog.askdirectory(parent=root, title=TITLES[kind], initialdir=d, mustexist=True)
    elif kind == "save":
        name = os.path.basename(start) if start and not os.path.isdir(os.path.expanduser(start)) else "fixed_std.sgy"
        p = filedialog.asksaveasfilename(parent=root, title=TITLES[kind], initialdir=d, initialfile=name,
                                         defaultextension=".sgy", filetypes=[("SEG-Y files", "*.sgy *.segy")])
    else:
        pats = SEGY_PATTERNS if kind == "file" else NAV_PATTERNS
        p = filedialog.askopenfilename(parent=root, title=TITLES[kind], initialdir=d,
                                       filetypes=[("SEG-Y files" if kind == "file" else "Navigation files", pats),
                                                  ("All files", "*.*")])
    root.destroy()
    print(json.dumps({"path": os.path.normpath(p) if p else None}))


def pick(kind: str, start: str | None = None) -> str | None:
    if kind not in KINDS:
        raise ValueError(f"unknown picker kind {kind!r}")
    if sys.platform == "darwin":
        return _mac(kind, start)
    if sys.platform.startswith("win"):
        return _tk_subprocess(kind, start)
    return _linux(kind, start)
