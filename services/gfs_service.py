from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

from adapters.gfs_adapter import process_file


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.getenv(
        "WEATHER_DATA_ROOT",
        str(BASE_DIR / "data"),
    )
)

RUN_FILE_PATTERN = re.compile(
    r"^(?P<prefix>.+?)_f(?P<start>\d{3})"
    r"(?:_f(?P<end>\d{3}))?"
    r"\.(?P<ext>grib2|grib|grb)$",
    re.IGNORECASE,
)

LEGACY_META_KEYS = {
    "png_url",
    "png_urls",
    "png_files",
    "binary_layer",
    "binary_layers",
    "binary_files",
    "binary_urls",
    "grid_files",
    "grid_urls",
    "diff_3km",
    "diff_1km",
    "resolution_variants",
    "resolution_options",
}


def _normalize_data_type(data_type: str | None) -> str:
    value = str(data_type or "").strip().upper()
    return "ECMWF" if value in {"EC", "ECMWF", "IFS"} else "GFS"


def _data_dir(data_type: str) -> Path:
    return DATA_ROOT / _normalize_data_type(data_type)


def _wait_dir(data_type: str) -> Path:
    return _data_dir(data_type) / "wait_process"


def _sorted_unique(paths: list[Path]) -> list[Path]:
    unique: dict[str, Path] = {}

    for path in paths:
        if not path.exists() or not path.is_file():
            continue

        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)

        unique[key] = path

    return sorted(
        unique.values(),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _list_files(
    data_type: str,
    patterns: tuple[str, ...],
) -> list[Path]:
    folders = [
        _wait_dir(data_type),
        _data_dir(data_type),
    ]

    for folder in folders:
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    files: list[Path] = []

    for folder in folders:
        for pattern in patterns:
            files.extend(
                path
                for path in folder.rglob(pattern)
                if ".adapter_staging" not in path.parts
            )

    return _sorted_unique(files)


def _list_grib_files(data_type: str) -> list[Path]:
    return _list_files(
        data_type,
        (
            "*.grib",
            "*.grb",
            "*.grib2",
        ),
    )


def _list_meta_files(data_type: str) -> list[Path]:
    return _list_files(
        data_type,
        ("*.meta.json",),
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return None

    return value if isinstance(value, dict) else None


def _write_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def _to_static_url(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(
            DATA_ROOT.resolve()
        )
        return "/data/" + str(relative).replace(
            "\\",
            "/",
        )
    except ValueError:
        return f"/data/{path.name}"


def _meta_is_current(
    meta: dict[str, Any] | None,
) -> bool:
    if not isinstance(meta, dict):
        return False

    if meta.get("schema_version") != "2.0":
        return False

    if meta.get("image_format") != "webp":
        return False

    if any(key in meta for key in LEGACY_META_KEYS):
        return False

    layers = meta.get("variable_layers")

    if not isinstance(layers, dict) or not layers:
        return False

    return any(
        isinstance(layer, dict)
        and isinstance(layer.get("frames"), list)
        and bool(layer.get("frames"))
        for layer in layers.values()
    )


def _parse_run_file(path: Path) -> dict[str, Any]:
    """
    支持以下文件名：

    ecmwf_20260707_12z_f000.grib2
    ecmwf_20260707_12z_f003.grib2
    ecmwf_realtime_20260707_12z_f000_f024.grib2
    gfs_20260707_18z_f006.grib2
    """

    match = RUN_FILE_PATTERN.match(path.name)

    if not match:
        return {
            "run_key": path.stem.lower(),
            "prefix": path.stem,
            "start_hour": 0,
            "end_hour": 0,
            "extension": (
                path.suffix.lower().lstrip(".")
                or "grib2"
            ),
            "path": path,
        }

    start_hour = int(match.group("start"))
    end_text = match.group("end")
    end_hour = (
        int(end_text)
        if end_text is not None
        else start_hour
    )

    return {
        "run_key": match.group("prefix").lower(),
        "prefix": match.group("prefix"),
        "start_hour": start_hour,
        "end_hour": end_hour,
        "extension": match.group("ext").lower(),
        "path": path,
    }


def _select_latest_run_files(
    data_type: str,
) -> list[dict[str, Any]]:
    """
    选择最新起报批次的所有 GRIB 文件。

    例如下面 9 个文件会被识别为同一个 ECMWF 批次：

    ecmwf_20260707_12z_f000.grib2
    ecmwf_20260707_12z_f003.grib2
    ...
    ecmwf_20260707_12z_f024.grib2
    """

    grib_files = _list_grib_files(data_type)

    if not grib_files:
        return []

    groups: dict[str, list[dict[str, Any]]] = {}

    for path in grib_files:
        item = _parse_run_file(path)
        groups.setdefault(
            str(item["run_key"]),
            [],
        ).append(item)

    latest_run_key = max(
        groups,
        key=lambda run_key: max(
            item["path"].stat().st_mtime
            for item in groups[run_key]
        ),
    )

    items = groups[latest_run_key]

    items.sort(
        key=lambda item: (
            int(item["start_hour"]),
            int(item["end_hour"]),
            item["path"].name,
        )
    )

    return items


def _ensure_single_file_meta(
    item: dict[str, Any],
    data_type: str,
) -> dict[str, Any]:
    source_file = Path(item["path"])

    meta_file = source_file.with_name(
        f"{source_file.name}.meta.json"
    )

    meta = (
        _read_json(meta_file)
        if meta_file.exists()
        else None
    )

    need_parse = (
        not meta_file.exists()
        or meta_file.stat().st_mtime
        < source_file.stat().st_mtime
        or not _meta_is_current(meta)
    )

    if need_parse:
        meta = process_file(
            str(source_file),
            data_type=data_type,
        )

    if not isinstance(meta, dict):
        raise ValueError(
            f"无法生成元数据：{source_file.name}"
        )

    return meta


def _frame_sort_key(
    frame: dict[str, Any],
) -> tuple[int, str]:
    try:
        forecast_hour = int(
            frame.get("forecast_hour", 0)
        )
    except (
        TypeError,
        ValueError,
    ):
        forecast_hour = 0

    return (
        forecast_hour,
        str(frame.get("valid_time") or ""),
    )


def _normalize_frames(
    frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique: dict[
        tuple[Any, Any, Any],
        dict[str, Any],
    ] = {}

    for frame in frames:
        if not isinstance(frame, dict):
            continue

        url = frame.get("url")

        if not url:
            continue

        forecast_hour = frame.get("forecast_hour")
        valid_time = frame.get("valid_time")

        # 同一时效可能同时来自“聚合文件”和“单时效文件”。
        # 优先按预报时效与有效时间去重，避免前端出现重复帧。
        key = (
            forecast_hour,
            valid_time,
        )

        # 极少数旧元数据没有时效和时间时，才使用 URL 区分。
        if forecast_hour is None and not valid_time:
            key = (
                "url",
                url,
            )

        if key not in unique:
            unique[key] = frame

    normalized = sorted(
        unique.values(),
        key=_frame_sort_key,
    )

    for index, frame in enumerate(normalized):
        frame["index"] = index

        try:
            forecast_hour = int(
                frame.get(
                    "forecast_hour",
                    index,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            forecast_hour = index

        frame["forecast_hour"] = forecast_hour

        frame["forecast_label"] = (
            frame.get("forecast_label")
            or f"F{forecast_hour:03d}"
        )

    return normalized


def _merge_run_metas(
    source: str,
    run_items: list[dict[str, Any]],
    metas: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    将同一批次多个单时效文件合并成一个多帧产品。

    这样前端可收到：

    frames = [
        F000,
        F003,
        F006,
        ...
        F024,
    ]
    """

    if not metas:
        raise ValueError(
            f"{source} 当前批次没有有效元数据。"
        )

    merged = copy.deepcopy(metas[0])

    merged_layers: dict[
        str,
        dict[str, Any],
    ] = {}

    variable_options: dict[
        str,
        dict[str, Any],
    ] = {}

    for item, meta in zip(
        run_items,
        metas,
    ):
        options = (
            meta.get("variable_options")
            or []
        )

        for option in options:
            if not isinstance(option, dict):
                continue

            variable_key = str(
                option.get("key") or ""
            )

            if (
                variable_key
                and variable_key
                not in variable_options
            ):
                variable_options[
                    variable_key
                ] = copy.deepcopy(option)

        source_layers = (
            meta.get("variable_layers")
            or {}
        )

        for (
            variable_key,
            source_layer,
        ) in source_layers.items():
            if not isinstance(
                source_layer,
                dict,
            ):
                continue

            if variable_key not in merged_layers:
                merged_layers[
                    variable_key
                ] = copy.deepcopy(source_layer)

                merged_layers[
                    variable_key
                ]["frames"] = []

            frames = copy.deepcopy(
                source_layer.get("frames")
                or []
            )

            if len(frames) == 1:
                frame = frames[0]

                try:
                    current_hour = int(
                        frame.get(
                            "forecast_hour",
                            0,
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    current_hour = 0

                filename_hour = int(
                    item["start_hour"]
                )

                if (
                    current_hour == 0
                    and filename_hour != 0
                ):
                    frame[
                        "forecast_hour"
                    ] = filename_hour

                    frame[
                        "forecast_label"
                    ] = (
                        f"F{filename_hour:03d}"
                    )

            merged_layers[
                variable_key
            ]["frames"].extend(frames)

    if not merged_layers:
        raise ValueError(
            f"{source} 当前批次没有可展示变量。"
        )

    for (
        variable_key,
        layer,
    ) in merged_layers.items():
        frames = _normalize_frames(
            layer.get("frames") or []
        )

        layer["frames"] = frames

        layer["times"] = [
            frame.get("valid_time")
            for frame in frames
        ]

        layer["forecast_hours"] = [
            frame.get("forecast_hour")
            for frame in frames
        ]

        layer["forecast_labels"] = [
            frame.get("forecast_label")
            for frame in frames
        ]

        layer["image_urls"] = [
            frame.get("url")
            for frame in frames
        ]

        layer["image_url"] = (
            layer["image_urls"][0]
            if layer["image_urls"]
            else layer.get("image_url")
        )

        if layer["times"]:
            if len(layer["times"]) == 1:
                layer["time"] = (
                    layer["times"][0]
                )
            else:
                layer["time"] = (
                    f"{layer['times'][0]} "
                    f"至 {layer['times'][-1]}"
                )

    default_variable = str(
        merged.get("default_variable")
        or next(iter(merged_layers))
    )

    if (
        default_variable
        not in merged_layers
    ):
        default_variable = next(
            iter(merged_layers)
        )

    default_layer = merged_layers[
        default_variable
    ]

    start_hour = min(
        int(item["start_hour"])
        for item in run_items
    )

    end_hour = max(
        int(item["end_hour"])
        for item in run_items
    )

    prefix = str(
        run_items[0]["prefix"]
    )

    extension = str(
        run_items[0]["extension"]
    )

    if start_hour != end_hour:
        merged_file_name = (
            f"{prefix}_"
            f"f{start_hour:03d}_"
            f"f{end_hour:03d}."
            f"{extension}"
        )
    else:
        merged_file_name = (
            run_items[0]["path"].name
        )

    if not variable_options:
        variable_options = {
            variable_key: {
                "key": variable_key,
                "label": (
                    layer.get("label")
                    or layer.get("element")
                    or variable_key
                ),
                "unit": layer.get("unit", ""),
                "varType": layer.get(
                    "varType",
                    "generic",
                ),
                "productCategory": layer.get(
                    "productCategory",
                    "数值预报产品",
                ),
            }
            for (
                variable_key,
                layer,
            ) in merged_layers.items()
        }

    merged.update(
        {
            "schema_version": "2.0",
            "source": source,
            "business_type": source,
            "data_type": source,
            "file_name": merged_file_name,
            "default_variable": default_variable,
            "variable_options": list(
                variable_options.values()
            ),
            "variable_layers": merged_layers,
            "extent": default_layer.get(
                "extent"
            ),
            "bbox": default_layer.get(
                "bbox"
            ),
            "grid": default_layer.get(
                "grid"
            ),
            "resolution": default_layer.get(
                "resolution"
            ),
            "run_time": default_layer.get(
                "issue_time"
            ),
            "image_format": "webp",
            "image_url": default_layer.get(
                "image_url"
            ),
            "image_urls": default_layer.get(
                "image_urls",
                [],
            ),
            "render_mode": "webp",
        }
    )

    weather_info = copy.deepcopy(
        merged.get("weather_info")
        or {}
    )

    weather_info.update(
        {
            "source": source,
            "file": merged_file_name,
            "file_name": merged_file_name,
            "element": default_layer.get(
                "element"
            ),
            "time": default_layer.get(
                "time"
            ),
            "level": default_layer.get(
                "level"
            ),
            "range": default_layer.get(
                "range"
            ),
            "resolution": default_layer.get(
                "resolution"
            ),
            "grid": (
                default_layer.get("gridText")
                or (
                    default_layer.get("grid")
                    or {}
                ).get("text")
            ),
            "unit": default_layer.get(
                "unit"
            ),
            "missing": default_layer.get(
                "missingText"
            ),
            "status": "解析成功",
            "quality": default_layer.get(
                "quality"
            ),
            "mainVariable": default_variable,
            "image_format": "webp",
            "image_url": default_layer.get(
                "image_url"
            ),
            "image_urls": default_layer.get(
                "image_urls",
                [],
            ),
            "render_mode": "webp",
        }
    )

    merged["weather_info"] = (
        weather_info
    )

    return merged


def _ensure_latest_meta(
    data_type: str,
) -> tuple[
    Path | None,
    dict[str, Any] | None,
    Path | None,
]:
    source = _normalize_data_type(
        data_type
    )

    run_items = _select_latest_run_files(
        source
    )

    if not run_items:
        meta_files = _list_meta_files(
            source
        )

        if not meta_files:
            return None, None, None

        return (
            meta_files[0],
            _read_json(meta_files[0]),
            None,
        )

    metas = [
        _ensure_single_file_meta(
            item,
            source,
        )
        for item in run_items
    ]

    merged_meta = _merge_run_metas(
        source,
        run_items,
        metas,
    )

    prefix = str(
        run_items[0]["prefix"]
    )

    merged_meta_file = (
        Path(run_items[0]["path"])
        .with_name(
            f"{prefix}.merged.meta.json"
        )
    )

    _write_json(
        merged_meta_file,
        merged_meta,
    )

    display_source_file = Path(
        str(
            merged_meta.get("file_name")
            or Path(
                run_items[0]["path"]
            ).name
        )
    )

    return (
        merged_meta_file,
        merged_meta,
        display_source_file,
    )


def get_display_data(
    data_type: str | None = "GFS",
) -> dict[str, Any]:
    """
    GFS / ECMWF 展示接口。

    返回：
    - 最新起报批次；
    - 同一批次所有时效；
    - WebP frames；
    - compact meta 2.0；
    - 不返回 PNG、float32、binary、3km 或 1km。
    """

    source = _normalize_data_type(
        data_type
    )

    (
        meta_file,
        meta,
        source_file,
    ) = _ensure_latest_meta(source)

    if not meta:
        return {
            "business_type": source,
            "data_type": source,
            "source": source,
            "status": "no_data",
            "message": (
                f"data/{source} 目录下"
                "暂无可展示的 GRIB/GRIB2 数据。"
            ),
            "image_format": "webp",
            "render_mode": "webp",
            "variable_options": [],
            "variable_layers": {},
        }

    default_variable = meta.get(
        "default_variable"
    )

    variable_layers = (
        meta.get("variable_layers")
        or {}
    )

    default_layer = (
        variable_layers.get(
            default_variable
        )
        or next(
            iter(
                variable_layers.values()
            ),
            {},
        )
    )

    file_name = (
        source_file.name
        if source_file is not None
        else meta.get("file_name")
    )

    return {
        "business_type": source,
        "data_type": source,
        "source": source,
        "status": "ok",
        "message": (
            f"{source} 数据读取成功"
        ),
        "file_name": file_name,
        "meta_url": (
            _to_static_url(meta_file)
            if meta_file is not None
            else None
        ),
        "schema_version": meta.get(
            "schema_version",
            "2.0",
        ),
        "run_time": meta.get(
            "run_time"
        ),
        "extent": meta.get(
            "extent"
        ),
        "bbox": meta.get(
            "bbox"
        ),
        "grid": meta.get(
            "grid"
        ),
        "resolution": meta.get(
            "resolution"
        ),
        "default_variable": (
            default_variable
        ),
        "variable_options": meta.get(
            "variable_options",
            [],
        ),
        "variable_layers": (
            variable_layers
        ),
        "weather_info": meta.get(
            "weather_info",
            {},
        ),
        "times": default_layer.get(
            "times",
            [],
        ),
        "forecast_hours": (
            default_layer.get(
                "forecast_hours",
                [],
            )
        ),
        "forecast_labels": (
            default_layer.get(
                "forecast_labels",
                [],
            )
        ),
        "frames": default_layer.get(
            "frames",
            [],
        ),
        "image_format": "webp",
        "image_url": (
            default_layer.get(
                "image_url"
            )
            or meta.get("image_url")
        ),
        "image_urls": (
            default_layer.get(
                "image_urls"
            )
            or meta.get(
                "image_urls",
                [],
            )
        ),
        "render_mode": "webp",
    }
