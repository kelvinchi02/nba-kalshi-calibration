"""Report which Kalshi settings are present, without printing secret values.

Usage:
    python scripts/check_secrets.py
"""
from pathlib import Path
import os

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"

# Settings that are secret: only report set/unset, never the value.
SECRET_KEYS = ["KALSHI_API_KEY_ID"]
# Settings that are not secret and are safe to show.
PUBLIC_KEYS = ["KALSHI_PRIVATE_KEY_PATH", "KALSHI_BASE_URL"]


def main() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
        print(f".env found at {ENV_PATH}")
    else:
        print(f".env NOT found at {ENV_PATH} (copy .env.example to .env)")

    for key in SECRET_KEYS:
        status = "set" if os.getenv(key, "").strip() else "NOT set"
        print(f"  {key}: {status}")

    for key in PUBLIC_KEYS:
        value = os.getenv(key, "").strip()
        print(f"  {key}: {value if value else 'NOT set'}")

    key_path_str = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
    if key_path_str:
        key_path = Path(key_path_str)
        if not key_path.is_absolute():
            key_path = REPO_ROOT / key_path
        if key_path.is_file():
            # Check the file looks like a PEM key without printing any of it.
            head = key_path.read_bytes()[:64]
            looks_pem = b"-----BEGIN" in head
            print(f"  private key file: exists ({'PEM header found' if looks_pem else 'no PEM header, check format'})")
        else:
            print("  private key file: does NOT exist")

    has_id = bool(os.getenv("KALSHI_API_KEY_ID", "").strip())
    has_key = bool(key_path_str) and key_path.is_file()
    if has_id and has_key:
        print("Authenticated requests: available")
    else:
        print("Authenticated requests: NOT available (public endpoints only)")


if __name__ == "__main__":
    main()
