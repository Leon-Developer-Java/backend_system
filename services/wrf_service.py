import json
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "WRF"
DATA_ROOT = DATA_DIR.parent
DEFAULT_RESOLUTION_KEY = "3km"


def _resolve_data_path(path: str) -> Path | None:
    text = str(path or "").replace("\\", "/")
    if not text:
        return None
    if text.startswith("/data/"):
        return DATA_ROOT / text.removeprefix("/data/")
    if text.startswith("data/"):
        return DATA_ROOT / text.removeprefix("data/")

    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    return DATA_DIR / candidate


def _existing_webp_files(meta_json: dict[str, Any]) -> list[str]:
    products = meta_json.get("resolution_products")
    if isinstance(products, dict) and products:
        key = meta_json.get("default_resolution") or DEFAULT_RESOLUTION_KEY
        product = products.get(key) or next(iter(products.values()))
        webp_files = [str(item).replace("\\", "/") for item in product.get("webp_files", [])]
    else:
        webp_files = [str(item).replace("\\", "/") for item in meta_json.get("webp_files", [])]
    return [
        item
        for item in webp_files
        if (resolved := _resolve_data_path(item)) is not None and resolved.exists()
    ]


def _has_resolution_products(meta_json: dict[str, Any]) -> bool:
    products = meta_json.get("resolution_products")
    if not isinstance(products, dict) or not products:
        return False
    return any(_existing_product_files(product) for product in products.values() if isinstance(product, dict))


def _existing_product_files(product: dict[str, Any]) -> list[str]:
    webp_files = [str(item).replace("\\", "/") for item in product.get("webp_files", [])]
    return [
        item
        for item in webp_files
        if (resolved := _resolve_data_path(item)) is not None and resolved.exists()
    ]


def _load_first_renderable_meta(meta_files: list[Path]) -> tuple[Path | None, dict[str, Any] | None, list[str]]:
    fallback: tuple[Path | None, dict[str, Any] | None, list[str]] = (None, None, [])

    loaded: list[tuple[Path, dict[str, Any], list[str]]] = []
    for meta_file in meta_files:
        with meta_file.open("r", encoding="utf-8") as file:
            meta_json = json.load(file)
        webp_files = _existing_webp_files(meta_json)
        loaded.append((meta_file, meta_json, webp_files))
        if fallback[0] is None:
            fallback = (meta_file, meta_json, webp_files)

    for meta_file, meta_json, webp_files in loaded:
        if webp_files and _has_resolution_products(meta_json):
            return meta_file, meta_json, webp_files

    for meta_file, meta_json, webp_files in loaded:
        if webp_files:
            return meta_file, meta_json, webp_files

    return fallback


def get_display_data() -> dict[str, Any]:
    meta_files = sorted(DATA_DIR.glob("*.meta.json"), key=lambda item: item.stat().st_mtime, reverse=True)

    meta_file, meta_json, webp_files = _load_first_renderable_meta(meta_files)

    return {
        "business_type": "WRF",
        "meta_file": str(meta_file).replace("\\", "/") if meta_file else None,
        "meta_json": meta_json,
        "webp": webp_files[0] if webp_files else None,
        "webp_files": webp_files,
        "default_resolution": meta_json.get("default_resolution") if meta_json else None,
        "resolution_products": meta_json.get("resolution_products") if meta_json else None,
    }
