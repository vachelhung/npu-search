"""Usage:
  python -m npusearch.cli index [--device NPU|GPU|CPU]
  python -m npusearch.cli search "query" [-k 5] [--path substring]
  python -m npusearch.cli status
"""
import argparse
import json
import sys

from . import index


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="npusearch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index"); p.add_argument("--device")
    p = sub.add_parser("search"); p.add_argument("query"); p.add_argument("-k", type=int, default=5)
    p.add_argument("--path"); p.add_argument("--device")
    sub.add_parser("status")
    a = ap.parse_args()

    if a.cmd == "index":
        print(json.dumps(index.build(a.device), ensure_ascii=False, indent=2))
    elif a.cmd == "status":
        print(json.dumps(index.status(), ensure_ascii=False, indent=2))
    else:
        for r in index.Searcher(a.device).search(a.query, a.k, a.path):
            snippet = r["text"].replace("\n", " ")[:160]
            print(f"{r['score']:.3f}  {r['path']}\n       {snippet}\n")


if __name__ == "__main__":
    main()
