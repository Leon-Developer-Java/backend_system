from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ERA5"
NODATA = -999999.0
WIND_PRODUCT = "10m_wind"
WIND_COMPONENTS = {"u": "u10", "v": "v10"}


class WindDisplayContractError(ValueError):
    def __init__(self, code: str, frame_index: int | None = None):
        super().__init__(code)
        self.code = code
        self.frame_index = frame_index


def _published_files(pattern: str) -> list[Path]:
    return [
        path
        for path in DATA_DIR.rglob(pattern)
        if ".adapter_staging" not in path.parts and path.is_file()
    ]


def get_display_data(variable: str | None = None, level_index: int = 0) -> dict[str, Any]:
    stored_meta = _latest_meta(allow_empty=True)
    wind_field = _display_wind_field(stored_meta)
    meta_json = deepcopy(stored_meta) if stored_meta else None
    if meta_json is not None:
        meta_json["wind_field"] = wind_field
    webp_files = sorted(_published_files("*.webp"), key=lambda item: item.stat().st_mtime, reverse=True)

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
        "wind_field": wind_field,
        "available_resolutions": meta_json.get("available_resolutions", []) if meta_json else [],
        "default_variable": selected,
        "times": meta_json.get("times", []) if meta_json else [],
        "extent": meta_json.get("extent") or meta_json.get("bbox") if meta_json else None,
        "grid": _grid_descriptor(meta_json, selected, layer) if meta_json and layer else None,
    }


def _latest_meta(allow_empty: bool = False) -> dict[str, Any] | None:
    meta_files = sorted(_published_files("*.meta.json"), key=lambda item: item.stat().st_mtime, reverse=True)
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
        candidates = sorted(_published_files("*.meta.json"), key=lambda item: item.stat().st_mtime, reverse=True)
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


def _unavailable_wind_field(
    reason: str,
    *,
    raw: dict[str, Any] | None = None,
    code: str | None = None,
    frame_index: int | None = None,
) -> dict[str, Any]:
    raw = raw or {}
    detail = raw.get("detail") if isinstance(raw.get("detail"), dict) else {}
    if code:
        detail = {"code": code}
        if frame_index is not None:
            detail["frame_index"] = frame_index
    components = raw.get("components")
    if not isinstance(components, dict):
        components = dict(WIND_COMPONENTS)
    return {
        "schema_version": str(raw.get("schema_version") or "1.0"),
        "available": False,
        "product": str(raw.get("product") or WIND_PRODUCT),
        "components": deepcopy(components),
        "reason": reason,
        "detail": deepcopy(detail),
    }


def _wind_asset(value: Any) -> tuple[str, Path]:
    if value is None:
        raise WindDisplayContractError("asset_url_missing")
    text = unquote(str(value).strip()).replace("\\", "/")
    if not text or "\x00" in text:
        raise WindDisplayContractError("asset_url_invalid")

    raw_path = Path(text)
    try:
        parsed = urlsplit(text)
    except ValueError as exc:
        raise WindDisplayContractError("asset_url_invalid") from exc
    if (
        parsed.netloc
        or parsed.query
        or parsed.fragment
        or (parsed.scheme and not raw_path.is_absolute())
    ):
        raise WindDisplayContractError("asset_url_not_local")

    marker = "/data/"
    marker_index = text.rfind(marker)
    if marker_index >= 0:
        relative_text = text[marker_index + len(marker):]
        candidate = DATA_DIR.parent / Path(relative_text)
    else:
        candidate = raw_path if raw_path.is_absolute() else DATA_DIR / raw_path

    try:
        root = DATA_DIR.resolve()
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        raise WindDisplayContractError("asset_path_unreadable") from exc
    if not resolved.is_relative_to(root):
        raise WindDisplayContractError("asset_path_outside_era5")
    if resolved.suffix.lower() != ".float32":
        raise WindDisplayContractError("asset_extension_invalid")
    relative = resolved.relative_to(root).as_posix()
    return f"/data/ERA5/{relative}", resolved


def _positive_int(value: Any, code: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WindDisplayContractError(code) from exc
    if number <= 0:
        raise WindDisplayContractError(code)
    return number


def _wind_speed_contract(
    meta: dict[str, Any],
    raw: dict[str, Any],
    times: list[str],
) -> tuple[str | None, dict[str, float] | None, list[str] | None]:
    speed_value = raw.get("speed_variable")
    range_value = raw.get("display_range")
    palette_value = raw.get("palette")
    if speed_value is None and range_value is None and palette_value is None:
        return None, None, None

    speed_variable = str(speed_value or "").strip()
    if not speed_variable:
        raise WindDisplayContractError("speed_variable_invalid")
    layers = meta.get("variable_layers")
    if not isinstance(layers, dict):
        raise WindDisplayContractError("speed_layer_missing")
    layer = layers.get(speed_variable)
    if not isinstance(layer, dict):
        raise WindDisplayContractError("speed_layer_missing")
    unit = str(layer.get("display_unit") or layer.get("unit") or "").strip().lower()
    if unit not in {"m/s", "m s-1", "m s**-1"}:
        raise WindDisplayContractError("speed_layer_unit_invalid")
    layer_times = layer.get("times")
    if not isinstance(layer_times, list) or [str(item) for item in layer_times] != times:
        raise WindDisplayContractError("speed_layer_times_invalid")

    if not isinstance(range_value, dict):
        raise WindDisplayContractError("speed_display_range_invalid")
    try:
        range_min = float(range_value.get("min"))
        range_max = float(range_value.get("max"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise WindDisplayContractError("speed_display_range_invalid") from exc
    if (
        not math.isfinite(range_min)
        or not math.isfinite(range_max)
        or range_min != 0.0
        or range_max <= range_min
    ):
        raise WindDisplayContractError("speed_display_range_invalid")

    if (
        not isinstance(palette_value, list)
        or len(palette_value) != 5
        or any(not isinstance(item, str) or not item.strip() for item in palette_value)
    ):
        raise WindDisplayContractError("speed_palette_invalid")
    layer_range = layer.get("display_range")
    layer_palette = layer.get("palette")
    if layer_range != range_value or layer_palette != palette_value:
        raise WindDisplayContractError("speed_style_mismatch")
    return speed_variable, {"min": range_min, "max": range_max}, list(palette_value)


def _display_wind_field(meta: dict[str, Any] | None) -> dict[str, Any]:
    raw = meta.get("wind_field") if isinstance(meta, dict) else None
    if not isinstance(raw, dict):
        return _unavailable_wind_field("not_provided")
    if raw.get("available") is not True:
        return _unavailable_wind_field(str(raw.get("reason") or "not_available"), raw=raw)

    try:
        components = raw.get("components")
        if (
            not isinstance(components, dict)
            or not str(components.get("u") or "")
            or not str(components.get("v") or "")
        ):
            raise WindDisplayContractError("components_invalid")

        grid = raw.get("grid")
        if not isinstance(grid, dict):
            raise WindDisplayContractError("grid_missing")
        width = _positive_int(grid.get("width"), "grid_width_invalid")
        height = _positive_int(grid.get("height"), "grid_height_invalid")
        extent = grid.get("extent")
        if not isinstance(extent, (list, tuple)) or len(extent) != 4:
            raise WindDisplayContractError("grid_extent_invalid")
        try:
            numeric_extent = [float(item) for item in extent]
        except (TypeError, ValueError, OverflowError) as exc:
            raise WindDisplayContractError("grid_extent_invalid") from exc
        if not all(math.isfinite(item) for item in numeric_extent):
            raise WindDisplayContractError("grid_extent_invalid")

        encoding = raw.get("encoding")
        if not isinstance(encoding, dict):
            raise WindDisplayContractError("encoding_missing")
        if str(encoding.get("dtype") or "").lower() != "float32":
            raise WindDisplayContractError("encoding_dtype_invalid")
        if str(encoding.get("byte_order") or "").lower() != "little":
            raise WindDisplayContractError("encoding_byte_order_invalid")
        if str(encoding.get("layout") or "").lower() != "component_separated":
            raise WindDisplayContractError("encoding_layout_invalid")
        bytes_per_value = _positive_int(
            encoding.get("bytes_per_value"),
            "encoding_bytes_per_value_invalid",
        )
        if bytes_per_value != 4:
            raise WindDisplayContractError("encoding_bytes_per_value_invalid")
        component_byte_length = width * height * bytes_per_value

        times_value = raw.get("times")
        frames_value = raw.get("frames")
        if not isinstance(times_value, list) or not isinstance(frames_value, list):
            raise WindDisplayContractError("frames_or_times_missing")
        times = [str(item) for item in times_value]
        if not times or len(frames_value) != len(times):
            raise WindDisplayContractError("frame_time_count_mismatch")

        speed_variable, display_range, palette = _wind_speed_contract(meta, raw, times)

        normalized_frames: list[dict[str, Any]] = []
        for frame_index, frame in enumerate(frames_value):
            if not isinstance(frame, dict):
                raise WindDisplayContractError("frame_invalid", frame_index)
            try:
                stored_index = int(frame.get("index"))
            except (TypeError, ValueError, OverflowError) as exc:
                raise WindDisplayContractError("frame_index_invalid", frame_index) from exc
            if stored_index != frame_index:
                raise WindDisplayContractError("frame_index_invalid", frame_index)
            if str(frame.get("time") or "") != times[frame_index]:
                raise WindDisplayContractError("frame_time_mismatch", frame_index)
            try:
                stored_byte_length = int(frame.get("component_byte_length"))
            except (TypeError, ValueError, OverflowError) as exc:
                raise WindDisplayContractError("frame_byte_length_invalid", frame_index) from exc
            if stored_byte_length != component_byte_length:
                raise WindDisplayContractError("frame_byte_length_invalid", frame_index)

            try:
                u_url, u_path = _wind_asset(frame.get("u_url"))
                v_url, v_path = _wind_asset(frame.get("v_url"))
            except WindDisplayContractError as exc:
                raise WindDisplayContractError(exc.code, frame_index) from exc
            for asset_path in (u_path, v_path):
                try:
                    if not asset_path.is_file():
                        raise WindDisplayContractError("asset_missing", frame_index)
                    if asset_path.stat().st_size != component_byte_length:
                        raise WindDisplayContractError("asset_byte_length_mismatch", frame_index)
                except WindDisplayContractError:
                    raise
                except (OSError, RuntimeError) as exc:
                    raise WindDisplayContractError("asset_unreadable", frame_index) from exc

            normalized_frame = deepcopy(frame)
            normalized_frame.update({
                "index": frame_index,
                "time": times[frame_index],
                "u_url": u_url,
                "v_url": v_url,
                "component_byte_length": component_byte_length,
            })
            normalized_frames.append(normalized_frame)

        descriptor = deepcopy(raw)
        descriptor.update({
            "schema_version": str(raw.get("schema_version") or "1.0"),
            "available": True,
            "product": str(raw.get("product") or WIND_PRODUCT),
            "components": deepcopy(components),
            "times": times,
            "grid": {**deepcopy(grid), "width": width, "height": height, "extent": numeric_extent},
            "encoding": {
                **deepcopy(encoding),
                "dtype": "float32",
                "byte_order": "little",
                "layout": "component_separated",
                "bytes_per_value": 4,
            },
            "frames": normalized_frames,
        })
        if speed_variable is not None:
            descriptor.update({
                "speed_variable": speed_variable,
                "display_range": display_range,
                "palette": palette,
            })
        return descriptor
    except WindDisplayContractError as exc:
        return _unavailable_wind_field(
            "display_contract_invalid",
            raw=raw,
            code=exc.code,
            frame_index=exc.frame_index,
        )
    except (OSError, RuntimeError, OverflowError):
        return _unavailable_wind_field(
            "display_contract_invalid",
            raw=raw,
            code="wind_validation_failed",
        )


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
