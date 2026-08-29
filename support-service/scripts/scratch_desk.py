"""Start a Bhasa instance that writes to a throwaway database.

Why this exists
---------------
Every verification round of this project has been run from a browser against
the live desk, because that is the only thing a browser could reach. The
standing instruction was to test against a scratch database; there was no way
to comply with it from a browser, so it was never followed, and the corpus
now holds several hundred rows of test traffic. Telling someone to use a
scratch database without giving them one to point at is not an instruction,
it is a wish.

This starts a second, identical instance on another port with its own empty
database. Nothing it writes touches the desk.

    python scripts/scratch_desk.py                # port 8090, fresh database
    python scripts/scratch_desk.py --port 9001
    python scripts/scratch_desk.py --keep         # keep rows from last run

The instance identifies itself: /v1/health reports "store": "scratch", and
the dashboard shows a SCRATCH marker in the top bar, so a tab open on the
scratch desk can never be mistaken for a tab open on the real one.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = Path(tempfile.gettempdir()) / "bhasa-scratch"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DIR / "support.db"),
        help=f"scratch database path (default: {DEFAULT_DIR / 'support.db'})",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the rows from the last run instead of starting empty",
    )
    args = parser.parse_args()

    db = Path(args.db).resolve()
    live = (SERVICE_ROOT / "data" / "store" / "support.db").resolve()
    # The one mistake this script must never make.
    if db == live:
        print(f"refusing to run: {db} is the live desk database", file=sys.stderr)
        return 2

    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists() and not args.keep:
        shutil.move(str(db), str(db.with_suffix(".db.previous")))
        print(f"previous scratch database moved to {db.with_suffix('.db.previous')}")

    env = dict(os.environ)
    env["SUPPORT_DB_PATH"] = str(db)
    env.pop("SUPPORT_ENV", None)  # a scratch desk is never production

    print(f"scratch desk   http://{args.host}:{args.port}")
    print(f"writing to     {db}")
    print(f"the real desk  {live}  (untouched)")
    sys.stdout.flush()

    os.environ.update(env)
    sys.path.insert(0, str(SERVICE_ROOT / "src"))
    import uvicorn

    uvicorn.run("himalaya_support.main:app", host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
