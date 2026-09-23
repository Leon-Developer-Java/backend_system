"""Authenticated read-only weather tools used by weather_agentscope."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from services import agent_analysis_tools, agent_db_tools


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
READ_OPERATIONS = {
    "query_weather_data",
    "db_inspect_data_assets",
    "db_list_parse_tasks",
    "db_get_file_assets",
    "db_query_assets",
    "db_get_latest_file_assets",
    "db_diagnose_parse_failures",
    "db_analyze_wrf_element_assets",
    "analyze_era5_timeseries",
    "analyze_radar_echo_change",
}


def _summarize(meta: dict[str, Any]) -> dict[str, Any]:
    variables = meta.get("variables") or []
    times = meta.get("times") or []
    images = meta.get("png_files") or meta.get("webp_files") or []
    return {
        "dataset_id": meta.get("dataset_id"),
        "data_type": meta.get("data_type"),
        "title": (meta.get("weather_info") or {}).get("product")
        or meta.get("data_type"),
        "variables": variables,
        "times": len(times) if isinstance(times, list) else times,
        "png_count": len(images),
        "png_files": images,
        "bbox": meta.get("bbox"),
    }


def list_datasets(data_type: str | None = None) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    for meta_path in DATA_ROOT.glob("**/*.meta.json"):
        try:
            datasets.append(_summarize(json.loads(meta_path.read_text("utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
    if not datasets:
        datasets = agent_db_tools.list_dataset_summaries(data_type)
    if data_type:
        key = data_type.strip().upper()
        datasets = [
            item
            for item in datasets
            if str(item.get("data_type") or "").upper() == key
        ]
    return {"count": len(datasets), "using_sample": False, "datasets": datasets}


def dispatch(operation: str, inputs: dict[str, Any]) -> dict[str, Any]:
    if operation not in READ_OPERATIONS:
        raise ValueError(f"unsupported read operation: {operation}")
    if operation == "query_weather_data":
        data = list_datasets(inputs.get("data_type") or None)
        return {
            **data,
            "status": "ok",
            "summary": f"已查询到 {len(data['datasets'])} 个数据集。",
            "records": data["datasets"],
        }
    if operation in {"analyze_era5_timeseries", "analyze_radar_echo_change"}:
        agent_analysis_tools.PROJECT_ROOT = DATA_ROOT.parent.parent
        agent_analysis_tools.DATA_ROOT = DATA_ROOT
        return agent_analysis_tools.dispatch(operation, inputs)
    return agent_db_tools.dispatch(operation, inputs)
