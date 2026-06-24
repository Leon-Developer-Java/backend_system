import base64
import json
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "Himawari"


def _as_posix(path: Path | None) -> str | None:
    return str(path).replace("\\", "/") if path else None


def _png_data_url(path: Path | None) -> str | None:
    if not path or not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _existing_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def _select_default_png(meta_json: dict[str, Any] | None, fallback: list[Path]) -> Path | None:
    if not meta_json:
        return fallback[0] if fallback else None

    for key in ("true_color", "natural_color", "water_vapor_enhanced"):
        for item in meta_json.get("composites", []):
            if item.get("key") == key and (path := _existing_path(item.get("png"))):
                return path

    for key in ("B13", "B14", "B08"):
        for item in meta_json.get("variables", []):
            if item.get("key") == key and (path := _existing_path(item.get("png"))):
                return path

    return fallback[0] if fallback else None


def _with_png_data_urls(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for item in items:
        path = _existing_path(item.get("png"))
        enriched.append({**item, "png_data_url": _png_data_url(path)})
    return enriched


def get_display_data() -> dict[str, Any]:
    meta_files = sorted(
        list(DATA_DIR.glob("*/*/meta/scene.meta.json")) + list(DATA_DIR.glob("*.meta.json")),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    png_files = sorted(
        list(DATA_DIR.glob("*/*/latlon/*.png")) + list(DATA_DIR.glob("*/*/composites/*.png")) + list(DATA_DIR.glob("*.png")),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )

    meta_json = None
    if meta_files:
        with meta_files[0].open("r", encoding="utf-8") as file:
            meta_json = json.load(file)

    png_path = _select_default_png(meta_json, png_files)
    variables = _with_png_data_urls(meta_json.get("variables", [])) if meta_json else []
    composites = _with_png_data_urls(meta_json.get("composites", [])) if meta_json else []

    return {
        "business_type": "Himawari",
        "meta_file": _as_posix(meta_files[0] if meta_files else None),
        "meta_json": meta_json,
        "extent": meta_json.get("extent") if meta_json else None,
        "grid": meta_json.get("grid") if meta_json else None,
        "variables": variables,
        "composites": composites,
        "png": _as_posix(png_path),
        "png_data_url": _png_data_url(png_path),
        "png_files": [_as_posix(path) for path in png_files if "latlon" in path.parts],
    }
