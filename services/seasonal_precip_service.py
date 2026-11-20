"""Latest seasonal precipitation display payload."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DATA_DIR = Path(
    os.getenv(
        "PRODUCT_DATA_ROOT",
        str(Path(__file__).resolve().parents[1] / "data"),
    )
).resolve() / "SeasonalPrecip"


def _meta_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(
        (path for path in DATA_DIR.rglob("*.meta.json") if ".adapter_staging" not in path.parts),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _read_meta(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _public_url(path: Path | str | None) -> str | None:
    if not path:
        return None
    text = str(path).replace("\\", "/")
    marker = "/data/"
    index = text.lower().find(marker)
    return text[index:] if index >= 0 else text


def get_display_data() -> dict[str, Any]:
    meta_path = next(iter(_meta_files()), None)
    meta = _read_meta(meta_path) if meta_path else None
    if not meta:
        return {
            "business_type": "SEASONAL_PRECIP",
            "data_type": "SEASONAL_PRECIP",
            "source": "SEASONAL_PRECIP",
            "status": "no_data",
            "message": "暂无已解析的季节性降水产品。",
            "image_format": "webp",
            "render_mode": "webp",
            "variable_options": [],
            "variable_layers": {},
        }

    layers = meta.get("variable_layers") if isinstance(meta.get("variable_layers"), dict) else {}
    default_variable = str(meta.get("default_variable") or next(iter(layers), ""))
    default_layer = layers.get(default_variable) if isinstance(layers.get(default_variable), dict) else next(iter(layers.values()), {})
    return {
        "business_type": "SEASONAL_PRECIP",
        "data_type": "SEASONAL_PRECIP",
        "source": "SEASONAL_PRECIP",
        "status": "ok",
        "message": "季节性降水产品读取成功",
        "file_name": meta.get("file_name"),
        "meta_url": _public_url(meta_path),
        "schema_version": meta.get("schema_version", "2.0"),
        "product": {
            "label": meta.get("product") or meta.get("file_name") or "季节性降水",
            "provider": meta.get("provider"),
        },
        "issue_time": meta.get("issue_time") or meta.get("run_time"),
        "run_time": meta.get("run_time"),
        "extent": meta.get("extent"),
        "bbox": meta.get("bbox"),
        "grid": meta.get("grid"),
        "resolution": meta.get("resolution"),
        "default_variable": default_variable,
        "variable_options": meta.get("variable_options", []),
        "variable_layers": layers,
        "times": default_layer.get("times", []),
        "forecast_hours": default_layer.get("forecast_hours", []),
        "forecast_labels": default_layer.get("forecast_labels", []),
        "frames": default_layer.get("frames", []),
        "image_format": "webp",
        "image_url": default_layer.get("image_url") or meta.get("image_url"),
        "image_urls": default_layer.get("image_urls") or meta.get("image_urls", []),
        "weather_info": meta.get("weather_info", {}),
        "render_mode": "webp",
    }
