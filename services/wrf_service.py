import json
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "WRF"


def get_display_data() -> dict[str, Any]:
    meta_files = sorted(DATA_DIR.glob("*.meta.json"), key=lambda item: item.stat().st_mtime, reverse=True)

    meta_json = None
    if meta_files:
        with meta_files[0].open("r", encoding="utf-8") as file:
            meta_json = json.load(file)

    webp_files: list[str] = []
    if meta_json:
        webp_files = [str(item).replace("\\", "/") for item in meta_json.get("webp_files", [])]

    return {
        "business_type": "WRF",
        "meta_file": str(meta_files[0]).replace("\\", "/") if meta_files else None,
        "meta_json": meta_json,
        "webp": webp_files[0] if webp_files else None,
        "webp_files": webp_files,
    }
