# PyInstaller recipe for the one-file desktop app:  pyinstaller packaging/segwise.spec
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

HERE = os.path.dirname(os.path.abspath(SPEC))  # noqa: F821 (PyInstaller global)

datas = collect_data_files("segwise") + collect_data_files("pyproj")
hidden = (collect_submodules("uvicorn") + [m for m in collect_submodules("segwise") if m != "segwise.mcp_server"]
          + ["tkinter", "tkinter.filedialog"])

a = Analysis(  # noqa: F821
    [os.path.join(HERE, "segwise_app.py")],
    datas=datas,
    hiddenimports=hidden,
    excludes=["mcp", "pytest", "IPython", "PyQt5", "PyQt6", "PySide2", "PySide6", "notebook", "segwise.mcp_server"],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz, a.scripts, a.binaries, a.datas, [],
    name="Segwise",
    console=True,  # the small window shows the address and closing it quits the app
    icon=os.path.join(HERE, "segwise.ico") if sys.platform.startswith("win") else None,
    upx=False,
    disable_windowed_traceback=False,
)
