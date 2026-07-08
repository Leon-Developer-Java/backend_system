from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ERA5"
DB_NAME = "era5_index.db"


def default_db_path() -> Path:
    return DATA_DIR / DB_NAME


def sync_meta(meta: dict[str, Any], db_path: str | Path | None = None) -> dict[str, int | str]:
    path = Path(db_path) if db_path else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with _connect(path) as conn:
        init_db(conn)
        result = _upsert_meta(conn, meta)
        conn.commit()

    return {"db_path": path.as_posix(), **result}


def list_datasets(
    *,
    keyword: str | None = None,
    variable: str | None = None,
    status: str | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    with _connect(db_path) as conn:
        init_db(conn)
        where, params = _dataset_filters(keyword, variable, status, time_start, time_end)
        total = conn.execute(f"SELECT COUNT(*) FROM era5_dataset d {where}", params).fetchone()[0]
        rows = conn.execute(
            f"""
            SELECT d.*
            FROM era5_dataset d
            {where}
            ORDER BY d.updated_at DESC, d.created_at DESC
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        ).fetchall()

    return {
        "items": [_dataset_row(row) for row in rows],
        "total": int(total),
        "limit": limit,
        "offset": offset,
    }


def get_dataset(dataset_id: str, db_path: str | Path | None = None) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        init_db(conn)
        dataset = conn.execute(
            "SELECT * FROM era5_dataset WHERE dataset_id = ? AND is_deleted = 0",
            (dataset_id,),
        ).fetchone()
        if dataset is None:
            return None

        variables = conn.execute(
            "SELECT * FROM era5_variable WHERE dataset_id = ? ORDER BY variable_name",
            (dataset_id,),
        ).fetchall()
        time_steps = conn.execute(
            "SELECT * FROM era5_time_step WHERE dataset_id = ? ORDER BY frame_index",
            (dataset_id,),
        ).fetchall()
        assets = conn.execute(
            """
            SELECT *
            FROM era5_layer_asset
            WHERE dataset_id = ?
            ORDER BY variable_name, resolution, frame_index
            """,
            (dataset_id,),
        ).fetchall()
        logs = conn.execute(
            """
            SELECT *
            FROM era5_parse_log
            WHERE dataset_id = ?
            ORDER BY id DESC
            LIMIT 200
            """,
            (dataset_id,),
        ).fetchall()

    return {
        **_dataset_row(dataset),
        "variables": [_variable_row(row) for row in variables],
        "time_steps": [_plain_row(row) for row in time_steps],
        "assets": [_plain_row(row) for row in assets],
        "logs": [_log_row(row) for row in logs],
    }


def list_assets(
    dataset_id: str,
    *,
    variable: str | None = None,
    resolution: str | None = None,
    limit: int = 500,
    offset: int = 0,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    where = ["dataset_id = ?"]
    params: list[Any] = [dataset_id]
    if variable:
        where.append("variable_name = ?")
        params.append(variable)
    if resolution:
        where.append("resolution = ?")
        params.append(resolution)
    where_sql = "WHERE " + " AND ".join(where)

    with _connect(db_path) as conn:
        init_db(conn)
        total = conn.execute(f"SELECT COUNT(*) FROM era5_layer_asset {where_sql}", params).fetchone()[0]
        rows = conn.execute(
            f"""
            SELECT *
            FROM era5_layer_asset
            {where_sql}
            ORDER BY variable_name, resolution, frame_index
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        ).fetchall()

    return {
        "items": [_plain_row(row) for row in rows],
        "total": int(total),
        "limit": limit,
        "offset": offset,
    }


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS era5_dataset (
            dataset_id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            source_file TEXT,
            meta_file TEXT,
            file_format TEXT,
            data_type TEXT,
            default_variable TEXT,
            time_start TEXT,
            time_end TEXT,
            lon_min REAL,
            lat_min REAL,
            lon_max REAL,
            lat_max REAL,
            nx INTEGER,
            ny INTEGER,
            variable_count INTEGER NOT NULL DEFAULT 0,
            time_count INTEGER NOT NULL DEFAULT 0,
            webp_count INTEGER NOT NULL DEFAULT 0,
            available_resolutions_json TEXT,
            status TEXT NOT NULL DEFAULT 'parsed',
            parse_error TEXT,
            file_size_bytes INTEGER,
            meta_summary_json TEXT,
            is_deleted INTEGER NOT NULL DEFAULT 0,
            delete_reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            deleted_at TEXT
        );

        CREATE TABLE IF NOT EXISTS era5_variable (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT NOT NULL,
            variable_name TEXT NOT NULL,
            name_cn TEXT,
            name_en TEXT,
            unit TEXT,
            dims_json TEXT,
            shape_json TEXT,
            min_value REAL,
            max_value REAL,
            mean_value REAL,
            std_value REAL,
            description TEXT,
            available_resolutions_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(dataset_id, variable_name),
            FOREIGN KEY(dataset_id) REFERENCES era5_dataset(dataset_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS era5_time_step (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT NOT NULL,
            frame_index INTEGER NOT NULL,
            valid_time TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(dataset_id, frame_index),
            FOREIGN KEY(dataset_id) REFERENCES era5_dataset(dataset_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS era5_layer_asset (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT NOT NULL,
            variable_name TEXT NOT NULL,
            resolution TEXT NOT NULL DEFAULT 'native',
            frame_index INTEGER NOT NULL,
            webp_url TEXT NOT NULL,
            width INTEGER,
            height INTEGER,
            lon_min REAL,
            lat_min REAL,
            lon_max REAL,
            lat_max REAL,
            min_value REAL,
            max_value REAL,
            mean_value REAL,
            std_value REAL,
            generated_at TEXT NOT NULL,
            UNIQUE(dataset_id, variable_name, resolution, frame_index),
            FOREIGN KEY(dataset_id) REFERENCES era5_dataset(dataset_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS era5_parse_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            message TEXT,
            detail_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(dataset_id) REFERENCES era5_dataset(dataset_id) ON DELETE CASCADE
        );
        """
    )


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _dataset_filters(
    keyword: str | None,
    variable: str | None,
    status: str | None,
    time_start: str | None,
    time_end: str | None,
) -> tuple[str, tuple[Any, ...]]:
    where = ["d.is_deleted = 0"]
    params: list[Any] = []
    if keyword:
        where.append("(d.file_name LIKE ? OR d.dataset_id LIKE ?)")
        like = f"%{keyword}%"
        params.extend([like, like])
    if variable:
        where.append(
            """
            EXISTS (
                SELECT 1
                FROM era5_variable v
                WHERE v.dataset_id = d.dataset_id AND v.variable_name = ?
            )
            """
        )
        params.append(variable)
    if status:
        where.append("d.status = ?")
        params.append(status)
    if time_start:
        where.append("(d.time_end IS NULL OR d.time_end >= ?)")
        params.append(time_start)
    if time_end:
        where.append("(d.time_start IS NULL OR d.time_start <= ?)")
        params.append(time_end)
    return "WHERE " + " AND ".join(where), tuple(params)


def _upsert_meta(conn: sqlite3.Connection, meta: dict[str, Any]) -> dict[str, int]:
    now = _now()
    dataset_id = str(meta.get("dataset_id") or "")
    if not dataset_id:
        raise ValueError("ERA5 meta is missing dataset_id.")

    source_file = str(meta.get("source_file") or "")
    source_path = Path(source_file) if source_file else None
    file_name = source_path.name if source_path else dataset_id
    file_size = source_path.stat().st_size if source_path and source_path.exists() else None
    times = [str(item) for item in meta.get("times") or []]
    bbox = _bbox(meta.get("bbox") or meta.get("extent"))
    variables = [item for item in meta.get("variables") or [] if isinstance(item, dict)]
    layers = meta.get("variable_layers") or {}
    webp_files = list(meta.get("webp_files") or [])
    status = str(meta.get("extra", {}).get("status") or "parsed")

    conn.execute(
        """
        INSERT INTO era5_dataset (
            dataset_id, file_name, source_file, meta_file, file_format, data_type,
            default_variable, time_start, time_end, lon_min, lat_min, lon_max, lat_max,
            nx, ny, variable_count, time_count, webp_count, available_resolutions_json,
            status, parse_error, file_size_bytes, meta_summary_json, is_deleted,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
        ON CONFLICT(dataset_id) DO UPDATE SET
            file_name = excluded.file_name,
            source_file = excluded.source_file,
            meta_file = excluded.meta_file,
            file_format = excluded.file_format,
            data_type = excluded.data_type,
            default_variable = excluded.default_variable,
            time_start = excluded.time_start,
            time_end = excluded.time_end,
            lon_min = excluded.lon_min,
            lat_min = excluded.lat_min,
            lon_max = excluded.lon_max,
            lat_max = excluded.lat_max,
            nx = excluded.nx,
            ny = excluded.ny,
            variable_count = excluded.variable_count,
            time_count = excluded.time_count,
            webp_count = excluded.webp_count,
            available_resolutions_json = excluded.available_resolutions_json,
            status = excluded.status,
            parse_error = excluded.parse_error,
            file_size_bytes = excluded.file_size_bytes,
            meta_summary_json = excluded.meta_summary_json,
            updated_at = excluded.updated_at
        """,
        (
            dataset_id,
            file_name,
            source_file,
            meta.get("meta_file"),
            meta.get("file_format"),
            meta.get("data_type"),
            meta.get("default_variable"),
            times[0] if times else None,
            times[-1] if times else None,
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3],
            _grid_size(meta, "width"),
            _grid_size(meta, "height"),
            len(variables),
            len(times),
            len(webp_files),
            _json(meta.get("available_resolutions") or []),
            status,
            meta.get("extra", {}).get("error"),
            file_size,
            _json(_summary(meta)),
            now,
            now,
        ),
    )

    conn.execute("DELETE FROM era5_variable WHERE dataset_id = ?", (dataset_id,))
    conn.execute("DELETE FROM era5_time_step WHERE dataset_id = ?", (dataset_id,))
    conn.execute("DELETE FROM era5_layer_asset WHERE dataset_id = ?", (dataset_id,))

    variable_count = _insert_variables(conn, dataset_id, variables, now)
    time_count = _insert_times(conn, dataset_id, times, now)
    asset_count = _insert_assets(conn, dataset_id, layers, bbox, now)
    log_count = _insert_logs(conn, dataset_id, layers, now)

    return {
        "datasets": 1,
        "variables": variable_count,
        "time_steps": time_count,
        "layer_assets": asset_count,
        "parse_logs": log_count,
    }


def _insert_variables(
    conn: sqlite3.Connection,
    dataset_id: str,
    variables: list[dict[str, Any]],
    now: str,
) -> int:
    rows = []
    for item in variables:
        name = str(item.get("name") or item.get("short_name") or item.get("raw_name") or "")
        if not name:
            continue
        stats = item.get("stats") or {}
        rows.append(
            (
                dataset_id,
                name,
                item.get("name_cn") or item.get("long_name") or name,
                item.get("long_name") or item.get("label") or name,
                item.get("unit") or item.get("display_unit") or "",
                _json(item.get("dims") or []),
                _json(item.get("shape") or []),
                _float_or_none(stats.get("min")),
                _float_or_none(stats.get("max")),
                _float_or_none(stats.get("mean")),
                _float_or_none(stats.get("std")),
                item.get("description") or item.get("long_name") or name,
                _json(item.get("available_resolutions") or []),
                now,
                now,
            )
        )

    conn.executemany(
        """
        INSERT INTO era5_variable (
            dataset_id, variable_name, name_cn, name_en, unit, dims_json, shape_json,
            min_value, max_value, mean_value, std_value, description,
            available_resolutions_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def _insert_times(conn: sqlite3.Connection, dataset_id: str, times: list[str], now: str) -> int:
    rows = [(dataset_id, index, value, now) for index, value in enumerate(times)]
    conn.executemany(
        """
        INSERT INTO era5_time_step (dataset_id, frame_index, valid_time, created_at)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def _insert_assets(
    conn: sqlite3.Connection,
    dataset_id: str,
    layers: dict[str, Any],
    default_bbox: tuple[float | None, float | None, float | None, float | None],
    now: str,
) -> int:
    rows = []
    for variable_name, layer in layers.items():
        if not isinstance(layer, dict):
            continue
        resolution_layers = layer.get("resolution_layers") or {"native": layer}
        for resolution, res_layer in resolution_layers.items():
            if not isinstance(res_layer, dict):
                continue
            bbox = _bbox(res_layer.get("extent") or layer.get("extent") or default_bbox)
            urls = res_layer.get("webp_urls") or res_layer.get("image_urls") or res_layer.get("png_urls") or []
            stats = res_layer.get("stats") or []
            for frame_index, url in enumerate(urls):
                stat = stats[frame_index] if frame_index < len(stats) and isinstance(stats[frame_index], dict) else {}
                rows.append(
                    (
                        dataset_id,
                        str(variable_name),
                        str(resolution),
                        frame_index,
                        str(url),
                        _int_or_none(res_layer.get("width") or layer.get("width")),
                        _int_or_none(res_layer.get("height") or layer.get("height")),
                        bbox[0],
                        bbox[1],
                        bbox[2],
                        bbox[3],
                        _float_or_none(stat.get("min")),
                        _float_or_none(stat.get("max")),
                        _float_or_none(stat.get("mean")),
                        _float_or_none(stat.get("std")),
                        now,
                    )
                )

    conn.executemany(
        """
        INSERT INTO era5_layer_asset (
            dataset_id, variable_name, resolution, frame_index, webp_url,
            width, height, lon_min, lat_min, lon_max, lat_max,
            min_value, max_value, mean_value, std_value, generated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def _insert_logs(conn: sqlite3.Connection, dataset_id: str, layers: dict[str, Any], now: str) -> int:
    rows = [(dataset_id, "parse", "parsed", "ERA5 meta synced to sqlite.", None, now)]
    quality_report = {}
    try:
        quality_report = conn.execute(
            "SELECT meta_summary_json FROM era5_dataset WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
    except sqlite3.Error:
        quality_report = {}

    if quality_report:
        summary = _loads(quality_report["meta_summary_json"], {})
        report = summary.get("quality_report")
        if isinstance(report, dict):
            rows.append(
                (
                    dataset_id,
                    "quality",
                    str(report.get("status") or "unknown"),
                    "ERA5 quality report.",
                    _json(report),
                    now,
                )
            )

    for variable_name, layer in layers.items():
        if not isinstance(layer, dict):
            continue
        for resolution, status in (layer.get("resolution_status") or {}).items():
            if not isinstance(status, dict):
                continue
            rows.append(
                (
                    dataset_id,
                    "resolution",
                    str(status.get("status") or "unknown"),
                    f"{variable_name}:{resolution}",
                    _json(status),
                    now,
                )
            )

    conn.executemany(
        """
        INSERT INTO era5_parse_log (dataset_id, stage, status, message, detail_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def _dataset_row(row: sqlite3.Row) -> dict[str, Any]:
    item = _plain_row(row)
    item["available_resolutions"] = _loads(item.pop("available_resolutions_json", None), [])
    item["meta_summary"] = _loads(item.pop("meta_summary_json", None), {})
    quality_report = item["meta_summary"].get("quality_report")
    item["quality_report"] = quality_report if isinstance(quality_report, dict) else None
    item["quality_status"] = item["quality_report"].get("status") if item["quality_report"] else None
    return item


def _variable_row(row: sqlite3.Row) -> dict[str, Any]:
    item = _plain_row(row)
    item["dims"] = _loads(item.pop("dims_json", None), [])
    item["shape"] = _loads(item.pop("shape_json", None), [])
    item["available_resolutions"] = _loads(item.pop("available_resolutions_json", None), [])
    return item


def _log_row(row: sqlite3.Row) -> dict[str, Any]:
    item = _plain_row(row)
    item["detail"] = _loads(item.pop("detail_json", None), None)
    return item


def _plain_row(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _summary(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "weather_info": meta.get("weather_info") or {},
        "levels": meta.get("levels") or [],
        "default_webp": meta.get("default_webp") or meta.get("default_png"),
        "quality_report": meta.get("quality_report") or meta.get("extra", {}).get("era5", {}).get("quality_report"),
    }


def _grid_size(meta: dict[str, Any], key: str) -> int | None:
    default_variable = meta.get("default_variable")
    layer = (meta.get("variable_layers") or {}).get(default_variable or "")
    if isinstance(layer, dict):
        return _int_or_none(layer.get(key))
    for item in (meta.get("variable_layers") or {}).values():
        if isinstance(item, dict):
            value = _int_or_none(item.get(key))
            if value is not None:
                return value
    return None


def _bbox(value: Any) -> tuple[float | None, float | None, float | None, float | None]:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return (
            _float_or_none(value[0]),
            _float_or_none(value[1]),
            _float_or_none(value[2]),
            _float_or_none(value[3]),
        )
    return (None, None, None, None)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
