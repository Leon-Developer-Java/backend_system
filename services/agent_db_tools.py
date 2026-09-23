"""Read-only database tools for the Agent.

These tools let the Agent see the same uploaded/parsed data catalog that the
system backend uses, instead of relying only on files scanned from data/.
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.engine import Engine, Row


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from DB.config import create_database_engine  # noqa: E402
from DB.repository import row_to_dict  # noqa: E402
from DB.schema import DETAIL_TABLES, public_info  # noqa: E402


SUPPORTED_TYPES = tuple(DETAIL_TABLES.keys())


@lru_cache(maxsize=1)
def _engine() -> Engine:
    return create_database_engine()


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _limit(value: Any, default: int = 20, maximum: int = 100) -> int:
    number = _as_int(value, default)
    return max(1, min(number, maximum))


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _normalize_type(data_type: str | None) -> str:
    text = str(data_type or "").strip().upper()
    if text == "RADAR":
        return "RADAR"
    return text


def _row_dict(row: Row | None) -> dict[str, Any] | None:
    return row_to_dict(row) if row is not None else None


def _compact_file(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "file_uuid": row.get("file_uuid"),
        "data_type": row.get("data_type"),
        "original_file_name": row.get("original_file_name"),
        "parse_status": row.get("parse_status"),
        "parse_error": row.get("parse_error"),
        "meta_path": row.get("meta_path"),
        "default_webp_url": row.get("default_webp_url"),
        "webp_count": row.get("webp_count"),
        "adapter_name": row.get("adapter_name"),
        "create_time": row.get("create_time"),
        "parse_finished_at": row.get("parse_finished_at"),
    }


def list_dataset_summaries(data_type: str | None = None) -> list[dict[str, Any]]:
    """Return parsed datasets summarized from DB detail tables."""
    normalized = _normalize_type(data_type)
    summaries: list[dict[str, Any]] = []
    with _engine().connect() as conn:
        table_items = DETAIL_TABLES.items()
        if normalized:
            table = DETAIL_TABLES.get(normalized)
            table_items = [(normalized, table)] if table is not None else []
        for dtype, detail_table in table_items:
            asset_count = conn.execute(
                select(func.count()).select_from(detail_table)
            ).scalar_one()
            if not asset_count:
                continue
            file_count = conn.execute(
                select(func.count(func.distinct(detail_table.c.file_uuid))).select_from(detail_table)
            ).scalar_one()
            elements = conn.execute(
                select(detail_table.c.element_key, func.count().label("count"))
                .group_by(detail_table.c.element_key)
                .order_by(func.count().desc(), detail_table.c.element_key.asc())
                .limit(20)
            ).all()
            times = conn.execute(
                select(func.min(detail_table.c.valid_time), func.max(detail_table.c.valid_time))
            ).first()
            summaries.append(
                {
                    "dataset_id": f"db_{dtype.lower()}",
                    "data_type": dtype,
                    "title": f"{dtype} 入库解析资产",
                    "variables": [row.element_key for row in elements if row.element_key],
                    "times": int(file_count or 0),
                    "asset_count": int(asset_count or 0),
                    "png_count": int(asset_count or 0),
                    "time_start": times[0].isoformat() + "Z" if times and times[0] else "",
                    "time_end": times[1].isoformat() + "Z" if times and times[1] else "",
                    "source": "DB/weather.db",
                }
            )
    return summaries


def db_inspect_data_assets() -> dict[str, Any]:
    """Inspect parse tasks and parsed asset rows in the shared DB."""
    with _engine().connect() as conn:
        task_total = conn.execute(select(func.count()).select_from(public_info)).scalar_one()
        status_rows = conn.execute(
            select(public_info.c.data_type, public_info.c.parse_status, func.count().label("count"))
            .where(public_info.c.is_deleted.is_(False))
            .group_by(public_info.c.data_type, public_info.c.parse_status)
            .order_by(public_info.c.data_type.asc(), public_info.c.parse_status.asc())
        ).all()
        latest_rows = conn.execute(
            select(public_info)
            .where(public_info.c.is_deleted.is_(False))
            .order_by(public_info.c.create_time.desc(), public_info.c.id.desc())
            .limit(8)
        ).all()

        detail_counts: list[dict[str, Any]] = []
        for dtype, detail_table in DETAIL_TABLES.items():
            asset_count = conn.execute(select(func.count()).select_from(detail_table)).scalar_one()
            if not asset_count:
                continue
            file_count = conn.execute(
                select(func.count(func.distinct(detail_table.c.file_uuid))).select_from(detail_table)
            ).scalar_one()
            elements = conn.execute(
                select(detail_table.c.element_key, func.count().label("count"))
                .group_by(detail_table.c.element_key)
                .order_by(func.count().desc(), detail_table.c.element_key.asc())
                .limit(12)
            ).all()
            detail_counts.append(
                {
                    "data_type": dtype,
                    "asset_count": int(asset_count or 0),
                    "file_count": int(file_count or 0),
                    "top_elements": [
                        {"element_key": row.element_key, "count": int(row.count or 0)}
                        for row in elements
                    ],
                }
            )

    status_summary = [
        {"data_type": row.data_type, "parse_status": row.parse_status, "count": int(row.count or 0)}
        for row in status_rows
    ]
    latest = [_compact_file(_row_dict(row) or {}) for row in latest_rows]

    lines = [f"DB 中共有 {int(task_total or 0)} 条上传/采集任务。"]
    for item in status_summary:
        lines.append(f"- {item['data_type']}：{item['parse_status']} {item['count']} 条")
    if detail_counts:
        lines.append("已入库可展示资产：")
        for item in detail_counts:
            element_text = "、".join(e["element_key"] for e in item["top_elements"][:6] if e["element_key"])
            lines.append(
                f"- {item['data_type']}：{item['file_count']} 个文件，{item['asset_count']} 条资产"
                + (f"，要素 {element_text}" if element_text else "")
            )
    else:
        lines.append("当前 detail 资产表为空，说明还没有成功解析出的可展示资产。")

    return {
        "status": "ok",
        "summary": "\n".join(lines),
        "data": {
            "task_total": int(task_total or 0),
            "status_summary": status_summary,
            "detail_counts": detail_counts,
            "latest_tasks": latest,
        },
        "image_url": None,
    }


def db_list_parse_tasks(
    data_type: str | None = None,
    parse_status: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    normalized = _normalize_type(data_type)
    conditions = [public_info.c.is_deleted.is_(False)]
    if normalized:
        conditions.append(func.upper(public_info.c.data_type) == normalized)
    if parse_status:
        conditions.append(public_info.c.parse_status == str(parse_status).strip().lower())

    with _engine().connect() as conn:
        total = conn.execute(select(func.count()).select_from(public_info).where(*conditions)).scalar_one()
        rows = conn.execute(
            select(public_info)
            .where(*conditions)
            .order_by(public_info.c.create_time.desc(), public_info.c.id.desc())
            .limit(_limit(limit, 20, 100))
        ).all()

    tasks = [_compact_file(_row_dict(row) or {}) for row in rows]
    title = "解析任务"
    if normalized:
        title += f"（{normalized}）"
    if parse_status:
        title += f"（状态 {parse_status}）"
    lines = [f"{title}共 {int(total or 0)} 条，返回最近 {len(tasks)} 条。"]
    for item in tasks[:10]:
        lines.append(
            f"- {item['original_file_name']}：{item['parse_status']}，"
            f"图像 {item.get('webp_count') or 0} 张，file_uuid={item['file_uuid']}"
        )
        if item.get("parse_error"):
            lines.append(f"  失败原因：{item['parse_error']}")
    return {"status": "ok", "summary": "\n".join(lines), "tasks": tasks, "image_url": None}


def db_get_file_assets(file_uuid: str, limit: int = 50) -> dict[str, Any]:
    file_uuid = str(file_uuid or "").strip()
    if not file_uuid:
        return {"status": "need_params", "summary": "需要提供 file_uuid。", "image_url": None}

    with _engine().connect() as conn:
        file_row = conn.execute(
            select(public_info).where(public_info.c.file_uuid == file_uuid, public_info.c.is_deleted.is_(False))
        ).first()
        file_item = _row_dict(file_row)
        if file_item is None:
            return {"status": "error", "summary": f"DB 中未找到 file_uuid={file_uuid}。", "image_url": None}
        dtype = _normalize_type(file_item.get("data_type"))
        detail_table = DETAIL_TABLES.get(dtype)
        asset_rows = []
        if detail_table is not None:
            asset_rows = conn.execute(
                select(detail_table)
                .where(detail_table.c.file_uuid == file_uuid)
                .order_by(detail_table.c.is_default.desc(), detail_table.c.element_key.asc(), detail_table.c.id.asc())
                .limit(_limit(limit, 50, 200))
            ).all()

    assets = [_row_dict(row) or {} for row in asset_rows]
    lines = [
        f"{file_item.get('original_file_name')}：{file_item.get('data_type')}，解析状态 {file_item.get('parse_status')}。",
        f"meta：{file_item.get('meta_path') or '无'}",
        f"默认图：{file_item.get('default_webp_url') or '无'}",
        f"返回资产 {len(assets)} 条。",
    ]
    for asset in assets[:10]:
        lines.append(
            f"- {asset.get('element_key')}，状态 {asset.get('asset_status')}，"
            f"webp={asset.get('webp_url') or '无'}"
        )
    return {"status": "ok", "summary": "\n".join(lines), "file": file_item, "assets": assets, "image_url": None}


def db_get_latest_file_assets(
    data_type: str | None = None,
    parse_status: str = "success",
    limit: int = 50,
) -> dict[str, Any]:
    normalized = _normalize_type(data_type) or "WRF"
    conditions = [
        public_info.c.is_deleted.is_(False),
        func.upper(public_info.c.data_type) == normalized,
    ]
    if parse_status:
        conditions.append(public_info.c.parse_status == str(parse_status).strip().lower())
    with _engine().connect() as conn:
        row = conn.execute(
            select(public_info)
            .where(*conditions)
            .order_by(public_info.c.create_time.desc(), public_info.c.id.desc())
            .limit(1)
        ).first()
    file_item = _row_dict(row)
    if file_item is None:
        return {
            "status": "error",
            "summary": f"未找到最新的 {normalized} {parse_status or ''} 文件记录。",
            "image_url": None,
        }
    result = db_get_file_assets(str(file_item.get("file_uuid") or ""), limit=limit)
    result["summary"] = f"最新 {normalized} 文件资产：\n" + str(result.get("summary") or "")
    return result


def db_query_assets(
    data_type: str,
    element_key: str | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    limit: int = 30,
) -> dict[str, Any]:
    normalized = _normalize_type(data_type)
    detail_table = DETAIL_TABLES.get(normalized)
    if detail_table is None:
        return {
            "status": "error",
            "summary": f"不支持的数据类型：{data_type}；可选 {', '.join(SUPPORTED_TYPES)}。",
            "image_url": None,
        }

    conditions = [detail_table.c.asset_status == "ready"]
    if element_key:
        conditions.append(func.upper(detail_table.c.element_key) == str(element_key).strip().upper())
    start_dt = _parse_dt(time_start)
    end_dt = _parse_dt(time_end)
    if start_dt:
        conditions.append(detail_table.c.valid_time >= start_dt)
    if end_dt:
        conditions.append(detail_table.c.valid_time <= end_dt)

    with _engine().connect() as conn:
        rows = conn.execute(
            select(
                detail_table,
                public_info.c.original_file_name.label("original_file_name"),
                public_info.c.parse_status.label("file_parse_status"),
                public_info.c.default_webp_url.label("file_default_webp_url"),
            )
            .join(public_info, public_info.c.file_uuid == detail_table.c.file_uuid)
            .where(and_(*conditions), public_info.c.is_deleted.is_(False))
            .order_by(detail_table.c.valid_time.asc(), detail_table.c.file_uuid.asc(), detail_table.c.id.asc())
            .limit(_limit(limit, 30, 200))
        ).all()

    assets = []
    for row in rows:
        item = dict(row._mapping)
        assets.append({key: (value.isoformat() + "Z" if isinstance(value, datetime) else value) for key, value in item.items()})

    lines = [f"DB 查询到 {len(assets)} 条 {normalized} 可用资产。"]
    for asset in assets[:12]:
        lines.append(
            f"- {asset.get('original_file_name')} / {asset.get('element_key')} / "
            f"{asset.get('valid_time') or '无时间'} / {asset.get('webp_url') or '无图像'}"
        )
    if not assets:
        lines.append("没有命中资产，请检查数据类型、要素名或时间范围。")
    return {"status": "ok", "summary": "\n".join(lines), "assets": assets, "image_url": None}


def db_analyze_wrf_element_assets(
    element_key: str,
    file_uuid: str | None = None,
    latest_only: bool = False,
    limit: int = 200,
) -> dict[str, Any]:
    element = str(element_key or "").strip().upper()
    if not element:
        return {"status": "need_params", "summary": "需要提供 WRF 要素名，例如 T2/RAINC/U10/V10。", "image_url": None}

    selected_file_uuid = str(file_uuid or "").strip()
    if latest_only and not selected_file_uuid:
        latest = db_list_parse_tasks(data_type="WRF", parse_status="success", limit=1)
        tasks = latest.get("tasks") if isinstance(latest.get("tasks"), list) else []
        if tasks and isinstance(tasks[0], dict):
            selected_file_uuid = str(tasks[0].get("file_uuid") or "")

    query = db_query_assets(data_type="WRF", element_key=element, limit=limit)
    assets = [item for item in query.get("assets", []) if isinstance(item, dict)]
    if selected_file_uuid:
        assets = [item for item in assets if str(item.get("file_uuid") or "") == selected_file_uuid]
    if not assets:
        scope = f"file_uuid={selected_file_uuid} 的 " if selected_file_uuid else ""
        return {"status": "error", "summary": f"未查询到 {scope}WRF {element} 可用资产。", "image_url": None}

    by_resolution: dict[str, int] = {}
    by_status: dict[str, int] = {}
    files: dict[str, str] = {}
    image_count = 0
    for asset in assets:
        resolution = str(asset.get("resolution_key") or "unknown")
        by_resolution[resolution] = by_resolution.get(resolution, 0) + 1
        status = str(asset.get("asset_status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        file_id = str(asset.get("file_uuid") or "")
        if file_id:
            files[file_id] = str(asset.get("original_file_name") or "")
        if asset.get("webp_url"):
            image_count += 1

    records = [{"resolution_key": key, "asset_count": count} for key, count in sorted(by_resolution.items())]
    status_records = [{"asset_status": key, "asset_count": count} for key, count in sorted(by_status.items())]
    file_scope = "最新文件" if latest_only else (f"文件 {selected_file_uuid}" if selected_file_uuid else "全部已入库文件")
    lines = [
        f"WRF {element} 资产分析完成，范围：{file_scope}。",
        f"共 {len(assets)} 条资产，涉及 {len(files)} 个文件，图像 URL {image_count} 个。",
        "按分辨率统计：" + "；".join(f"{item['resolution_key']}={item['asset_count']}" for item in records),
        "按状态统计：" + "；".join(f"{item['asset_status']}={item['asset_count']}" for item in status_records),
        "数据来源：DB/weather.db -> wrf_info/public_info。",
    ]
    return {
        "status": "ok",
        "summary": "\n".join(lines),
        "records": records,
        "status_records": status_records,
        "assets": assets[:50],
        "sources": [{"database": "DB/weather.db", "tables": ["public_info", "wrf_info"]}],
        "image_url": assets[0].get("webp_url") if assets and assets[0].get("webp_url") else None,
    }


def db_diagnose_parse_failures(data_type: str | None = None, limit: int = 50) -> dict[str, Any]:
    normalized = _normalize_type(data_type)
    conditions = [public_info.c.is_deleted.is_(False), public_info.c.parse_status == "failed"]
    if normalized:
        conditions.append(func.upper(public_info.c.data_type) == normalized)
    with _engine().connect() as conn:
        rows = conn.execute(
            select(public_info.c.data_type, public_info.c.parse_error, func.count().label("count"))
            .where(*conditions)
            .group_by(public_info.c.data_type, public_info.c.parse_error)
            .order_by(func.count().desc())
            .limit(_limit(limit, 50, 100))
        ).all()
        latest_rows = conn.execute(
            select(public_info)
            .where(*conditions)
            .order_by(public_info.c.parse_finished_at.desc(), public_info.c.id.desc())
            .limit(10)
        ).all()
    groups = [
        {
            "data_type": row.data_type,
            "parse_error": row.parse_error or "未知失败原因",
            "count": int(row.count or 0),
            "retryable": _failure_retryable(str(row.parse_error or "")),
            "suggestion": _failure_suggestion(str(row.parse_error or "")),
        }
        for row in rows
    ]
    latest = [_compact_file(_row_dict(row) or {}) for row in latest_rows]
    total = sum(item["count"] for item in groups)
    lines = [f"解析失败诊断完成，共 {total} 条失败记录。"]
    if normalized:
        lines[0] = f"{normalized} 解析失败诊断完成，共 {total} 条失败记录。"
    for item in groups[:8]:
        retry_text = "建议重试" if item["retryable"] else "不建议直接重试"
        lines.append(f"- {item['data_type']}：{item['count']} 条；原因：{item['parse_error']}；{retry_text}；{item['suggestion']}")
    if not groups:
        lines.append("当前没有匹配的失败解析任务。")
    return {
        "status": "ok",
        "summary": "\n".join(lines),
        "groups": groups,
        "tasks": latest,
        "sources": [{"database": "DB/weather.db", "table": "public_info"}],
        "image_url": None,
    }


def _failure_retryable(error: str) -> bool:
    text = error.lower()
    if any(token in text for token in ("timeout", "temporary", "busy", "locked", "permission", "超时")):
        return True
    if any(token in error for token in ("未找到可渲染", "格式", "不支持")):
        return False
    if any(token in text for token in ("unsupported", "not supported", "missing variable")):
        return False
    return False


def _failure_suggestion(error: str) -> str:
    text = error.lower()
    if "未找到可渲染" in error:
        return "优先检查文件变量名、adapter 支持变量和产品类型，补充映射后再重试。"
    if "timeout" in text or "超时" in error:
        return "可在确认原始文件存在后重试，并检查 adapter 超时配置。"
    if "permission" in text or "权限" in error:
        return "检查原始文件和输出目录权限后再重试。"
    return "先检查原始文件是否存在、数据类型是否正确、adapter 是否支持该产品。"


def label_for(name: str) -> str | None:
    return {
        "db_inspect_data_assets": "查询 DB 数据资产中",
        "db_list_parse_tasks": "查询 DB 解析任务中",
        "db_get_file_assets": "查询 DB 文件资产中",
        "db_query_assets": "查询 DB 资产明细中",
        "db_get_latest_file_assets": "查询最新文件资产中",
        "db_analyze_wrf_element_assets": "分析 WRF 要素资产中",
        "db_diagnose_parse_failures": "诊断解析失败任务中",
    }.get(name)


def dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "db_inspect_data_assets":
        return db_inspect_data_assets()
    if name == "db_list_parse_tasks":
        return db_list_parse_tasks(**(args or {}))
    if name == "db_get_file_assets":
        return db_get_file_assets(**(args or {}))
    if name == "db_get_latest_file_assets":
        return db_get_latest_file_assets(**(args or {}))
    if name == "db_query_assets":
        return db_query_assets(**(args or {}))
    if name == "db_analyze_wrf_element_assets":
        return db_analyze_wrf_element_assets(**(args or {}))
    if name == "db_diagnose_parse_failures":
        return db_diagnose_parse_failures(**(args or {}))
    return {"status": "error", "summary": f"未知 DB 工具：{name}", "image_url": None}


def list_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "db_inspect_data_assets",
                "description": "查看系统共享数据库中的上传/解析任务、解析状态分布、已入库资产数量和主要要素。用户问数据库、入库数据、真实数据资产时优先调用。",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "db_list_parse_tasks",
                "description": "按数据类型或解析状态列出 DB 中的上传/解析任务，包含 file_uuid、原始文件名、解析状态、失败原因、meta 路径和默认图。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_type": {"type": "string", "description": "可选，ERA5/GFS/ECMWF/Radar/WRF/Himawari/CMA/FY3"},
                        "parse_status": {"type": "string", "description": "可选，pending/running/success/failed"},
                        "limit": {"type": "integer", "description": "返回数量，默认 20，最大 100"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "db_get_file_assets",
                "description": "根据 file_uuid 查询某个文件的 DB 入库记录和资产明细。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_uuid": {"type": "string", "description": "public_info.file_uuid"},
                        "limit": {"type": "integer", "description": "资产返回数量，默认 50"},
                    },
                    "required": ["file_uuid"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "db_query_assets",
                "description": "查询某类已入库资产明细，可按要素和时间过滤，返回图像 URL、时间、范围和文件来源。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_type": {"type": "string", "description": "必填，ERA5/GFS/ECMWF/Radar/WRF/Himawari/CMA/FY3"},
                        "element_key": {"type": "string", "description": "可选，要素名，如 T2、RAINC、DBZH"},
                        "time_start": {"type": "string", "description": "可选，开始时间 ISO 字符串"},
                        "time_end": {"type": "string", "description": "可选，结束时间 ISO 字符串"},
                        "limit": {"type": "integer", "description": "返回数量，默认 30，最大 200"},
                    },
                    "required": ["data_type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "db_get_latest_file_assets",
                "description": "查询某类数据最新解析成功文件的资产明细。适合用户说最新 WRF、刚上传的 WRF、最新那个文件有哪些要素/图像。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_type": {"type": "string", "description": "可选，默认 WRF"},
                        "parse_status": {"type": "string", "description": "可选，默认 success"},
                        "limit": {"type": "integer", "description": "资产返回数量，默认 50"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "db_analyze_wrf_element_assets",
                "description": "稳定分析 WRF 指定要素资产，返回资产数量、涉及文件数量、分辨率分布、状态分布和数据来源。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "element_key": {"type": "string", "description": "必填，WRF 要素名，如 T2、RAINC、RAINNC、U10、V10、PSFC、PM10、PM2_5_DRY"},
                        "file_uuid": {"type": "string", "description": "可选，限定某个文件"},
                        "latest_only": {"type": "boolean", "description": "可选，只分析最新解析成功 WRF 文件"},
                        "limit": {"type": "integer", "description": "最多读取资产数量，默认 200"},
                    },
                    "required": ["element_key"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "db_diagnose_parse_failures",
                "description": "诊断解析失败任务，按失败原因聚合，判断是否建议重试并给出处理建议。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_type": {"type": "string", "description": "可选，按数据类型过滤，如 Radar/WRF/GFS"},
                        "limit": {"type": "integer", "description": "返回失败原因分组数量，默认 50"},
                    },
                },
            },
        },
    ]
