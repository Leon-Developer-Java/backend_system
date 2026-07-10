import json
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ERA5"
NODATA = -999999.0


def get_display_data(variable: str | None = None, level_index: int = 0) -> dict[str, Any]:
    meta_json = _latest_meta(allow_empty=True)
    webp_files = sorted(DATA_DIR.glob("*.webp"), key=lambda item: item.stat().st_mtime, reverse=True)

    variables = _display_variables(meta_json)
    selected = _primary_variable(meta_json, variable) if meta_json else ""
    layer = _layer_for_variable(meta_json, selected) if meta_json else None
    first_image = _first_image(meta_json, layer, webp_files)

    return {
        "business_type": "ERA5",
        "meta_file": _path_string(_meta_path(meta_json)) if meta_json else None,
        "meta_json": meta_json,
        "webp": first_image,
        "image_url": first_image,
        "webp_files": _all_images(meta_json, webp_files),
        "variables": variables,
        "variable_options": meta_json.get("variable_options", variables) if meta_json else [],
        "variable_layers": meta_json.get("variable_layers", {}) if meta_json else {},
        "available_resolutions": meta_json.get("available_resolutions", []) if meta_json else [],
        "default_variable": selected,
        "times": meta_json.get("times", []) if meta_json else [],
        "extent": meta_json.get("extent") or meta_json.get("bbox") if meta_json else None,
        "grid": _grid_descriptor(meta_json, selected, layer) if meta_json and layer else None,
    }


def _latest_meta(allow_empty: bool = False) -> dict[str, Any] | None:
    meta_files = sorted(DATA_DIR.glob("*.meta.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for meta_file in meta_files:
        with meta_file.open("r", encoding="utf-8") as file:
            meta = json.load(file)
        if meta.get("variable_layers"):
            return meta
    if meta_files:
        with meta_files[0].open("r", encoding="utf-8") as file:
            return json.load(file)
    if allow_empty:
        return None
    raise ValueError("No ERA5 meta.json found. Parse an ERA5 file first.")


def _meta_path(meta: dict[str, Any] | None) -> Path | None:
    if not meta:
        return None
    value = meta.get("meta_file")
    if value:
        path = Path(str(value))
        if path.exists():
            return path
        fallback = DATA_DIR / path.name
        if fallback.exists():
            return fallback
    dataset_id = meta.get("dataset_id")
    if dataset_id:
        candidates = sorted(DATA_DIR.glob("*.meta.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        for candidate in candidates:
            try:
                with candidate.open("r", encoding="utf-8") as file:
                    if json.load(file).get("dataset_id") == dataset_id:
                        return candidate
            except Exception:
                continue
    return None


def _path_string(path: Path | None) -> str | None:
    return str(path).replace("\\", "/") if path else None


def _public_url(path: str | Path | None) -> str | None:
    if not path:
        return None
    normalized = str(path).replace("\\", "/")
    marker = "/data/"
    idx = normalized.rfind(marker)
    return normalized[idx:] if idx >= 0 else normalized


def _first_image(
    meta: dict[str, Any] | None,
    layer: dict[str, Any] | None,
    webp_files: list[Path],
) -> str | None:
    if layer:
        urls = layer.get("webp_urls") or layer.get("image_urls") or []
        if urls:
            return _public_url(urls[0])
    if meta and meta.get("default_webp"):
        return _public_url(meta.get("default_webp"))
    if webp_files:
        return _path_string(webp_files[0])
    return None


def _all_images(
    meta: dict[str, Any] | None,
    webp_files: list[Path],
) -> list[str]:
    result: list[str] = []
    if meta:
        for item in meta.get("webp_files") or []:
            public = _public_url(item)
            if public and public not in result:
                result.append(public)
        for layer in (meta.get("variable_layers") or {}).values():
            for field in ("webp_urls", "image_urls"):
                for item in layer.get(field) or []:
                    public = _public_url(item)
                    if public and public not in result:
                        result.append(public)
    for path in webp_files:
        value = _path_string(path)
        if value and value not in result:
            result.append(value)
    return result


def _display_variables(meta: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not meta:
        return []

    options = meta.get("variable_options")
    if isinstance(options, list) and options:
        return [
            {
                "name": item.get("name"),
                "label": item.get("label") or item.get("long_name") or item.get("name"),
                "unit": item.get("unit") or item.get("display_unit") or "",
            }
            for item in options
            if isinstance(item, dict) and item.get("name")
        ]

    result: list[dict[str, Any]] = []
    for item in meta.get("variables") or []:
        if isinstance(item, dict):
            name = item.get("name")
            if name:
                result.append({
                    "name": name,
                    "label": item.get("long_name") or item.get("label") or name,
                    "unit": item.get("unit") or item.get("units") or item.get("display_unit") or "",
                })
        elif item:
            result.append({"name": str(item), "label": str(item), "unit": ""})
    return result


def _primary_variable(meta: dict[str, Any] | None, requested: str | None = None) -> str:
    if requested:
        return str(requested)
    if not meta:
        return ""

    primary = meta.get("default_variable") or meta.get("extra", {}).get("default_variable")
    if not primary:
        primary = meta.get("extra", {}).get("era5", {}).get("default_variable")
    if primary:
        return str(primary)

    variables = _display_variables(meta)
    if variables:
        return str(variables[0]["name"])

    raise ValueError("No ERA5 variable found in meta.json.")


def _layer_for_variable(meta: dict[str, Any] | None, variable: str) -> dict[str, Any] | None:
    if not meta or not variable:
        return None
    layers = meta.get("variable_layers") or {}
    if variable in layers:
        return layers[variable]
    lowered = {str(key).lower(): value for key, value in layers.items()}
    return lowered.get(variable.lower())


def _grid_descriptor(meta: dict[str, Any], variable: str, layer: dict[str, Any]) -> dict[str, Any]:
    stats = layer.get("stats") or []
    first_stats = stats[0] if stats else {}
    webp_urls = layer.get("webp_urls") or layer.get("image_urls") or []
    image_urls = webp_urls
    image_url = _public_url(image_urls[0]) if image_urls else None
    return {
        "business_type": "ERA5",
        "dataset_id": meta.get("dataset_id"),
        "file": Path(str(meta.get("source_file", ""))).name,
        "variable": variable,
        "label": layer.get("label") or variable,
        "unit": layer.get("unit") or "",
        "width": int(layer.get("width") or 0),
        "height": int(layer.get("height") or 0),
        "extent": layer.get("extent") or meta.get("extent") or meta.get("bbox"),
        "min": first_stats.get("min", 0.0),
        "max": first_stats.get("max", 1.0),
        "mean": first_stats.get("mean", 0.0),
        "nodata": layer.get("nodata", NODATA),
        "webp_url": image_url,
        "image_url": image_url,
        "webp_urls": webp_urls,
        "image_urls": image_urls,
        "available_resolutions": layer.get("available_resolutions") or meta.get("available_resolutions") or [],
        "resolution_layers": layer.get("resolution_layers") or {},
        "resolution_status": layer.get("resolution_status") or {},
        "resolution": layer.get("resolution") or "",
        "times": layer.get("times") or meta.get("times") or [],
        "stats": stats,
        "variables": _display_variables(meta),
        "meta": {
            "source": "ERA5",
            "element": f"{layer.get('label') or variable} ({variable})",
            "unit": layer.get("unit") or "",
            "extent": layer.get("extent") or meta.get("extent") or meta.get("bbox"),
            "grid": f"{int(layer.get('width') or 0)} x {int(layer.get('height') or 0)}",
            "time": (layer.get("times") or meta.get("times") or [""])[0],
            "min": first_stats.get("min", 0.0),
            "max": first_stats.get("max", 1.0),
            "mean": first_stats.get("mean", 0.0),
        },
    }
