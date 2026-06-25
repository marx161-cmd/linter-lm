"""Command-line ingestion helper.

    python -m contextstore.cli add path/to/file.txt --tags code,spectreboard
    python -m contextstore.cli add path/to/lyrics.txt --tags lyrics --name "song name"
    python -m contextstore.cli list
    python -m contextstore.cli rm <file_id>

Reads files you point it at and writes them straight into the store -- no
model involved in deciding what gets ingested or how it's tagged, that's
entirely your call. This is the only intended way content enters the store.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from .contextstore import ContextStore


async def cmd_add(args: argparse.Namespace) -> None:
    store = ContextStore()
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    file = await store.ingest_file(args.path, tags=tags, name=args.name)
    print(f"ingested: {file.id}  name={file.name!r}  tags={file.tags}  chars={len(file.content)}")


async def cmd_list(args: argparse.Namespace) -> None:
    store = ContextStore()
    files = store.list_files()
    if not files:
        print("(empty)")
        return
    for f in files:
        print(f"{f.id}  {f.name!r}  tags={f.tags}  chars={len(f.content)}")


async def cmd_rm(args: argparse.Namespace) -> None:
    store = ContextStore()
    ok = store.delete(args.file_id)
    print("deleted" if ok else "no such file_id")


def main() -> None:
    parser = argparse.ArgumentParser(prog="contextstore")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="ingest a file")
    p_add.add_argument("path")
    p_add.add_argument("--tags", default="", help="comma-separated tags")
    p_add.add_argument("--name", default=None, help="display name (default: filename)")
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="list ingested files")
    p_list.set_defaults(func=cmd_list)

    p_rm = sub.add_parser("rm", help="delete a file by id")
    p_rm.add_argument("file_id")
    p_rm.set_defaults(func=cmd_rm)

    args = parser.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
