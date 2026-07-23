from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


def _first(*values, default=None):
    for value in values:
        if value is not None and value != "":
            return value
    return default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _bbox(value: Any) -> tuple[float | None, float | None, float | None, float | None]:
    if isinstance(value, dict):
        return (
            _float(value.get("west")),
            _float(value.get("south")),
            _float(value.get("east")),
            _float(value.get("north")),
        )
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return tuple(_float(item) for item in value[:4])
    return None, None, None, None


def _grid(layer: dict[str, Any]) -> tuple[int | None, int | None]:
    width = _int(_first(layer.get("width"), layer.get("grid_width")))
    height = _int(_first(layer.get("height"), layer.get("grid_height")))
    grid = layer.get("grid")
    if isinstance(grid, dict):
        width = width or _int(_first(grid.get("nx"), grid.get("width"), grid.get("cols")))
        height = height or _int(_first(grid.get("ny"), grid.get("height"), grid.get("rows")))
    if (not width or not height) and isinstance(grid, str):
        match = re.search(r"(\d+)\s*[xX×]\s*(\d+)", grid)
        if match:
            width = width or int(match.group(1))
            height = height or int(match.group(2))
    return width, height


def _path_value(value: Any, index: int = 0) -> Any:
    items = _as_list(value)
    if not items:
        return None
    return items[index] if index < len(items) else items[-1]


def _webp_values(layer: dict[str, Any]) -> list[str]:
    for key in ("webp_urls", "image_urls", "webp_files"):
        values = [str(item) for item in _as_list(layer.get(key)) if str(item).lower().endswith(".webp")]
        if values:
            return values
    for key in ("webp_url", "image_url", "webp"):
        value = layer.get(key)
        if isinstance(value, dict):
            values = _as_list(value.get("paths") or value.get("path"))
        else:
            values = _as_list(value)
        result = [str(item) for item in values if str(item).lower().endswith(".webp")]
        if result:
            return result
    frames = layer.get("frames")
    if isinstance(frames, list):
        result = []
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            value = _first(frame.get("webp_url"), frame.get("image_url"), frame.get("webp"))
            if value and str(value).lower().endswith(".webp"):
                result.append(str(value))
        return result
    return []


def _normalize_webp_url(value: str, product_root: Path, final_dir: Path) -> str | None:
    text = str(value or "").replace("\\", "/")
    if not text:
        return None
    if text.startswith("/data/"):
        candidate = product_root / text.removeprefix("/data/")
        if candidate.is_file():
            return text
    if text.startswith("data/"):
        candidate = product_root / text.removeprefix("data/")
        if candidate.is_file():
            return "/" + text
    candidate = Path(text)
    if candidate.is_absolute() and candidate.is_file():
        try:
            return "/data/" + candidate.resolve().relative_to(product_root.resolve()).as_posix()
        except ValueError:
            return None
    matches = list(final_dir.rglob(Path(text).name))
    if len(matches) == 1:
        return "/data/" + matches[0].resolve().relative_to(product_root.resolve()).as_posix()
    return None


def _parsed_path(value: Any, product_root: Path, final_dir: Path) -> str | None:
    if not value:
        return None
    text = str(value).replace("\\", "/")
    if text.startswith("/data/"):
        return text.removeprefix("/data/")
    if text.startswith("data/"):
        return text.removeprefix("data/")
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(product_root.resolve()).as_posix()
        except ValueError:
            return None
    matches = list(final_dir.rglob(candidate.name))
    if len(matches) == 1:
        return matches[0].resolve().relative_to(product_root.resolve()).as_posix()
    return text


def _specific_fields(data_type: str, meta: dict[str, Any], layer: dict[str, Any], valid_time: datetime | None) -> dict[str, Any]:
    extra = meta.get("extra") if isinstance(meta.get("extra"), dict) else {}
    if data_type == "ERA5":
        era5 = extra.get("era5") if isinstance(extra.get("era5"), dict) else {}
        return {
            "source_dataset": _first(era5.get("source_dataset"), meta.get("dataset_id")),
            "product_type": _first(era5.get("product_type"), meta.get("product")),
            "data_stream": _first(era5.get("data_stream"), layer.get("data_stream")),
            "step_type": _first(layer.get("stepType"), layer.get("step_type")),
            "grid_type": _first(era5.get("grid_type"), meta.get("grid_type")),
            "coordinate_system": _first(era5.get("coordinate_system"), "latitude_longitude"),
            "native_lon_resolution": _float(era5.get("native_lon_resolution")),
            "native_lat_resolution": _float(era5.get("native_lat_resolution")),
        }
    if data_type == "CMA":
        cma = extra.get("cma") if isinstance(extra.get("cma"), dict) else {}
        return {
            "product_type": cma.get("product_type"),
            "product_name": cma.get("product_name"),
            "data_time": valid_time,
            "native_resolution_lon": _float(cma.get("native_resolution_lon")),
            "native_resolution_lat": _float(cma.get("native_resolution_lat")),
        }
    if data_type == "WRF":
        return {
            "domain": _first(meta.get("domain"), layer.get("domain")),
            "forecast_reference_time": _datetime(_first(meta.get("forecast_reference_time"), meta.get("run_time"))),
            "forecast_hour": _int(_first(layer.get("forecast_hour"), meta.get("forecast_hour"))),
            "dx_m": _float(meta.get("dx_m")),
            "dy_m": _float(meta.get("dy_m")),
            "source_resolution": _first(meta.get("resolution"), layer.get("resolution")),
        }
    if data_type == "RADAR":
        details = meta.get("format_specific") if isinstance(meta.get("format_specific"), dict) else {}
        return {
            "radar_name": details.get("radar_name"),
            "station_code": details.get("station_code"),
            "radar_type": details.get("radar_type"),
            "institution": details.get("institution"),
            "product_code": _first(layer.get("productCode"), details.get("product_code")),
            "observed_at": valid_time,
            "observed_end_at": _datetime(details.get("observed_end_at")),
            "elevation": _float(details.get("elevation")),
        }
    if data_type in {"GFS", "ECMWF"}:
        run_time = _datetime(_first(meta.get("run_time"), layer.get("issue_time")))
        frame_index = _int(layer.get("_frame_index")) or 0
        common = {
            "run_time": run_time,
            "cycle_hour": run_time.hour if run_time else None,
            "forecast_hour": _int(_first(layer.get("forecast_hour"), _path_value(layer.get("forecast_hours"), frame_index))),
            "step_type": _first(layer.get("stepType"), layer.get("step_type")),
            "type_of_level": _first(layer.get("typeOfLevel"), layer.get("level_type")),
            "interpolation_method": layer.get("interpolation_method"),
        }
        if data_type == "GFS":
            common["product_category"] = layer.get("productCategory")
        else:
            common["stream"] = _first(meta.get("stream"), layer.get("stream"))
            common["product_class"] = _first(meta.get("product_class"), layer.get("product_class"))
        return common
    if data_type == "FY3":
        return {
            "scene_id": meta.get("scene_id"),
            "satellite": meta.get("satellite"),
            "instrument": _first(meta.get("instrument"), meta.get("sensor")),
            "band": _first(layer.get("band"), layer.get("name")),
            "wavelength": _float(layer.get("wavelength")),
            "source_resolution": _first(layer.get("resolution"), meta.get("resolution")),
            "file_role": layer.get("file_role"),
            "paired_file_uuid": None,
        }
    return {}


def build_asset_catalog(
    *,
    file_uuid: str,
    data_type: str,
    meta: dict[str, Any],
    product_root: Path,
    final_dir: Path,
) -> list[dict[str, Any]]:
    data_type = data_type.upper()
    actual_webps = sorted(path for path in final_dir.rglob("*.webp") if path.is_file())
    default_url = _first(meta.get("default_webp"), meta.get("image_url"))
    dataset_id = str(_first(meta.get("dataset_id"), file_uuid))
    meta_times = _as_list(meta.get("times"))
    west, south, east, north = _bbox(_first(meta.get("bbox"), meta.get("extent")))
    candidates: list[tuple[str, dict[str, Any], str, list[str]]] = []

    variable_layers = meta.get("variable_layers")
    if isinstance(variable_layers, dict):
        for element_key, layer in variable_layers.items():
            if not isinstance(layer, dict):
                continue
            resolutions = layer.get("resolution_layers")
            if isinstance(resolutions, dict) and resolutions:
                for resolution_key, resolution_layer in resolutions.items():
                    if isinstance(resolution_layer, dict):
                        candidates.append((str(element_key), {**layer, **resolution_layer}, str(resolution_key), _webp_values(resolution_layer)))
            else:
                candidates.append((str(element_key), layer, str(layer.get("resolution") or "native"), _webp_values(layer)))

    variables = meta.get("variables")
    if isinstance(variables, list):
        for variable in variables:
            if not isinstance(variable, dict):
                continue
            key = str(_first(variable.get("name"), variable.get("short_name"), variable.get("raw_name"), default="unknown"))
            resolutions = variable.get("resolution_layers")
            if isinstance(resolutions, dict) and resolutions:
                for resolution_key, resolution_layer in resolutions.items():
                    if isinstance(resolution_layer, dict):
                        paths = _webp_values(resolution_layer)
                        if paths:
                            candidates.append((key, {**variable, **resolution_layer}, str(resolution_key), paths))
            paths = _webp_values(variable)
            if paths:
                candidates.append((key, variable, "native", paths))

    seen: set[str] = set()
    assets: list[dict[str, Any]] = []
    for element_key, layer, resolution_key, paths in candidates:
        times = _as_list(_first(layer.get("times"), meta_times))
        stats = _as_list(_first(layer.get("step_stats"), layer.get("stats")))
        grid_paths = _as_list(_first(layer.get("grid_urls"), layer.get("float32_urls")))
        width, height = _grid(layer)
        layer_bbox = _bbox(_first(layer.get("bbox"), layer.get("extent")))
        if any(value is not None for value in layer_bbox):
            layer_west, layer_south, layer_east, layer_north = layer_bbox
        else:
            layer_west, layer_south, layer_east, layer_north = west, south, east, north
        for frame_index, raw_path in enumerate(paths):
            webp_url = _normalize_webp_url(raw_path, product_root, final_dir)
            if not webp_url or webp_url in seen:
                continue
            seen.add(webp_url)
            stat = _path_value(stats, frame_index)
            stat = stat if isinstance(stat, dict) else {}
            valid_time = _datetime(_path_value(times, frame_index))
            asset_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_uuid}:{element_key}:{resolution_key}:{frame_index}:{webp_url}"))
            asset = {
                "asset_uuid": asset_uuid,
                "file_uuid": file_uuid,
                "dataset_id": dataset_id,
                "element_key": element_key,
                "raw_element_name": _first(layer.get("raw_name"), layer.get("shortName"), element_key),
                "element_label": _first(layer.get("label"), layer.get("long_name"), layer.get("name_cn"), element_key),
                "element_kind": _first(layer.get("varType"), layer.get("category")),
                "raw_unit": _first(layer.get("rawUnit"), layer.get("raw_unit"), layer.get("unit")),
                "display_unit": _first(layer.get("displayUnit"), layer.get("display_unit"), layer.get("unit")),
                "level_type": _first(layer.get("typeOfLevel"), layer.get("level_type")),
                "level_value": str(_first(layer.get("level"), layer.get("level_value"), default="surface")),
                "valid_time": valid_time,
                "frame_index": frame_index,
                "resolution_key": resolution_key or "native",
                "grid_width": width,
                "grid_height": height,
                "bbox_west": layer_west,
                "bbox_south": layer_south,
                "bbox_east": layer_east,
                "bbox_north": layer_north,
                "parsed_data_path": _parsed_path(_path_value(grid_paths, frame_index), product_root, final_dir),
                "webp_url": webp_url,
                "min_value": _float(_first(stat.get("min"), layer.get("min"))),
                "max_value": _float(_first(stat.get("max"), layer.get("max"))),
                "mean_value": _float(_first(stat.get("mean"), layer.get("mean"))),
                "missing_ratio": _float(_first(stat.get("missing_ratio"), layer.get("missing_ratio"))),
                "is_default": bool(webp_url == default_url or element_key == meta.get("default_variable") and frame_index == 0),
                "asset_status": "ready",
                "extra_json": json.dumps({"source": "meta.json"}, ensure_ascii=False),
            }
            asset.update(_specific_fields(data_type, meta, {**layer, "_frame_index": frame_index}, valid_time))
            assets.append(asset)

    for path in actual_webps:
        webp_url = "/data/" + path.resolve().relative_to(product_root.resolve()).as_posix()
        if webp_url in seen:
            continue
        frame_match = re.search(r"(?:step|frame)(\d+)", path.stem, re.IGNORECASE)
        frame_index = int(frame_match.group(1)) if frame_match else 0
        known_elements = {str(asset["element_key"]) for asset in assets}
        element_key = next(
            (key for key in known_elements if path.stem.endswith(f"_{key}") or f"_{key}_step" in path.stem),
            None,
        )
        if element_key is None:
            element_match = re.search(r"[_\.]([A-Za-z][A-Za-z0-9_]{0,31})(?:_(?:step|frame)\d+)?$", path.stem)
            element_key = element_match.group(1) if element_match else "unknown"
        assets.append({
            "asset_uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_uuid}:{webp_url}")),
            "file_uuid": file_uuid,
            "dataset_id": dataset_id,
            "element_key": element_key,
            "raw_element_name": element_key,
            "element_label": element_key,
            "level_value": "surface",
            "frame_index": frame_index,
            "resolution_key": "native",
            "bbox_west": west,
            "bbox_south": south,
            "bbox_east": east,
            "bbox_north": north,
            "webp_url": webp_url,
            "is_default": bool(webp_url == default_url or not assets),
            "asset_status": "ready",
            "extra_json": json.dumps({"source": "filesystem_fallback"}, ensure_ascii=False),
            **_specific_fields(data_type, meta, {}, None),
        })
        seen.add(webp_url)
    return assets
