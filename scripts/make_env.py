"""Create .env for a first run: the example settings plus a freshly generated API key.

Run: python scripts/make_env.py
Standard library only, same on Windows, macOS and Linux. An existing .env is never overwritten.
The key is printed once, here; it is also in .env, which stays on this machine.
"""

import argparse
import json
import secrets
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", default="dev", help="name of the API client (default: dev)")
    parser.add_argument(
        "--scopes", default="chat,admin", help="comma-separated scopes (default: chat,admin)"
    )
    parser.add_argument("--output", type=Path, default=ROOT / ".env", help="file to create")
    parser.add_argument(
        "--fake-llm",
        action="store_true",
        help="point docker compose at the stand-in model (no Ollama or GPU needed)",
    )
    parser.add_argument(
        "--print-key-only", action="store_true", help="print only the key (for scripts)"
    )
    args = parser.parse_args()

    if args.output.exists():
        print(f"{args.output} already exists, left untouched", file=sys.stderr)
        return 1

    shutil.copyfile(ROOT / ".env.example", args.output)
    key = secrets.token_urlsafe(32)
    keys = [{"client": args.client, "key": key, "scopes": args.scopes.split(",")}]
    with args.output.open("a", encoding="utf-8") as env:
        env.write(f"\nAPI_KEYS={json.dumps(keys)}\n")
        if args.fake_llm:
            env.write("COMPOSE_LLM_URL=http://fake-llm:9000/v1\n")

    if args.print_key_only:
        print(key)
    else:
        print(f"Created {args.output}")
        print(f"API key for client '{args.client}': {key}")
        print("Send it as the X-API-Key header. It is stored only in .env.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
