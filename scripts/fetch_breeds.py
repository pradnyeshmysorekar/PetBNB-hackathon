"""Development-only utility to build static breed reference files.

Run once with: python scripts/fetch_breeds.py
The Streamlit app reads the resulting JSON files locally and never calls these APIs.
"""

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
load_dotenv(PROJECT_ROOT / ".env")
SOURCES = {
    "dog_breeds.json": ("https://api.thedogapi.com/v1/breeds", "THEDOGAPI_KEY"),
    "cat_breeds.json": ("https://api.thecatapi.com/v1/breeds", "THECATAPI_KEY"),
}


def fetch_breed_names(url: str, key_environment_variable: str) -> list[str]:
    api_key = os.getenv(key_environment_variable, "DEMO-API-KEY")
    response = requests.get(
        url,
        headers={"Accept": "application/json", "x-api-key": api_key},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected response format from {url}")
    names = {item["name"].strip() for item in payload if isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"].strip()}
    if not names:
        raise ValueError(f"No breed names received from {url}")
    return sorted(names, key=str.casefold)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for filename, (url, key_environment_variable) in SOURCES.items():
        names = fetch_breed_names(url, key_environment_variable)
        output_path = DATA_DIR / filename
        output_path.write_text(json.dumps(names, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {len(names)} breeds to {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
