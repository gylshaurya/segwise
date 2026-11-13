"""Entry point of the Windows app (Segwise.exe).

Double-clicking starts the local web app and opens the browser.
Run from a terminal with arguments, it behaves exactly like the `segwise` command,
e.g. `Segwise.exe scan survey.sgy` or `Segwise.exe inventory D:\\deliveries\\2026-11`.
"""
import multiprocessing
import os
import sys


def main() -> int:
    multiprocessing.freeze_support()  # batch checks use worker processes; required in a frozen app
    from segwise.cli import main as cli_main

    args = sys.argv[1:]
    if not args:
        args = ["ui", "--root", os.path.expanduser("~")]
    try:
        return cli_main(args)
    except SystemExit as e:
        return int(e.code or 0)
    except Exception as e:  # keep the window open long enough to read the error
        import traceback

        traceback.print_exc()
        if not sys.argv[1:]:
            input(f"\nsegwise could not start: {e}\nPress Enter to close this window.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
