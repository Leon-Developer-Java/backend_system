import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

WEATHER_BACKEND = os.getenv("WEATHER_BACKEND_BASE", "http://127.0.0.1:8002").rstrip("/")
BACKEND_SYSTEM_DIR = Path(os.getenv("BACKEND_SYSTEM_DIR", r"D:\xiazai\python\pythonproject\backend_system"))
PYTHON_BIN = os.getenv("PYTHON_BIN", sys.executable or "python")
DATA_DIR = BACKEND_SYSTEM_DIR / "data"
DOWNLOAD_SCRIPT = BACKEND_SYSTEM_DIR / "scripts" / "download_gfs_ecmwf.py"
HTTP_TIMEOUT = float(os.getenv("AGENT_HTTP_TIMEOUT", "60"))
MAX_SUBPROCESS_SECONDS = int(os.getenv("AGENT_SUBPROCESS_TIMEOUT", "1800"))

app = FastAPI(title="Weather Data Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5177", "http://127.0.0.1:5177",
        "http://localhost:5178", "http://127.0.0.1:5178",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class AgentChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    context: dict[str, Any] | None = None

def ndjson(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"

def text_event(value: str) -> str:
    return ndjson({"type": "text", "value": value})

def done_event() -> str:
    return ndjson({"type": "done"})


def image_event(url: str, caption: str = "") -> str:
    return ndjson({
        "type": "image",
        "url": url,
        "src": url,
        "caption": caption,
        "alt": caption,
        "urls": [url],
        "images": [url],
    })


def make_backend_asset_url(url: str) -> str:
    url = str(url or "").strip()
    if not url:
        return ""
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("/"):
        return WEATHER_BACKEND + url
    return WEATHER_BACKEND + "/" + url

def error_event(message: str) -> str:
    return ndjson({"type": "error", "message": message})

def tool_event(name: str, label: str, progress: int, result: str = "", status: str = "running") -> str:
    return ndjson({"type": "tool", "name": name, "label": label, "progress": progress, "result": result, "status": status})

def last_user_text(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content.strip()
    return ""

def normalize_source(value: str | None) -> str:
    text = str(value or "").upper()
    if "ECMWF" in text or re.search(r"\bEC\b", text) or "欧洲" in text or "IFS" in text:
        return "ECMWF"
    if "GFS" in text or "美国" in text:
        return "GFS"
    return "ECMWF"

def detect_intent(text: str) -> str:
    t = text.strip().lower()
    zh = text
    if any(k in text for k in ["帮助", "怎么用", "能做什么", "指令"]) or t in {"help", "/help"}:
        return "help"
    if any(k in text for k in ["下载", "更新", "拉取"]) or any(k in t for k in ["download", "update"]):
        return "download"
    if any(k in text for k in ["重新解析", "解析", "生成webp", "生成 webp"]) or any(k in t for k in ["parse", "render"]):
        return "download"
    if any(k in text for k in ["审计", "完整性", "缺失", "资源", "文件齐全"]) or any(k in t for k in ["audit", "missing"]):
        return "audit"
    if any(k in text for k in ["不显示", "显示失败", "报错", "诊断", "为什么"]) or any(k in t for k in ["diagnose", "error", "fail"]):
        return "diagnose"
    if "webp" in t or "二进制" in text or "png" in t:
        return "check_format"
    if any(k in text for k in ["报告", "总结", "日报"]) or "report" in t:
        return "report"
    if any(k in text for k in ["查询", "最新", "状态", "现在", "数据"]) or any(k in t for k in ["query", "status", "latest"]):
        return "query"
    if any(k in text for k in ["删除", "清理", "硬删除", "软删除"]):
        return "delete_guard"
    if any(k in zh for k in ["双源", "全部数据源", "所有数据源", "对比", "比较"]) or ("GFS" in text.upper() and ("ECMWF" in text.upper() or "EC" in text.upper())):
        return "compare_sources"

    if any(k in zh for k in ["生成图表", "图表", "出图", "展示图", "显示图", "画图"]) or "/生成图表" in t or "chart" in t or "image" in t:
        return "chart"

    if any(k in zh for k in ["调用模型", "运行模型", "跑模型", "模型推理"]) or "/调用模型" in t:
        return "call_model"

    return "chat"

def parse_lead_params(text: str, source: str) -> dict[str, Any]:
    upper = text.upper()
    lead_start = 0
    lead_end = 24 if source == "ECMWF" else 47
    lead_step = 3 if source == "ECMWF" else 1
    m = re.search(r"F\s*0*(\d{1,3})\s*[-_到至~]\s*F?\s*0*(\d{1,3})", upper)
    if m:
        lead_start, lead_end = int(m.group(1)), int(m.group(2))
    else:
        m2 = re.search(r"(?:到|至|前|未来)?\s*(\d{1,3})\s*(?:小时|H|HR|HOUR)", upper)
        if m2:
            lead_end = int(m2.group(1))
    m3 = re.search(r"(?:STEP|间隔|步长)\s*[=:：]?\s*(\d{1,2})", upper)
    if m3:
        lead_step = int(m3.group(1))
    return {
        "lead_start": lead_start,
        "lead_end": lead_end,
        "lead_step": lead_step,
        "overwrite": any(k in text.lower() for k in ["覆盖", "overwrite", "--overwrite"]),
        "insecure_ssl": any(k in text.lower() for k in ["insecure", "跳过ssl", "跳过 ssl", "ssl"]),
    }

async def fetch_json(url: str) -> tuple[int, dict[str, Any] | None, str]:
    # trust_env=False prevents local 127.0.0.1 requests from being routed through system proxy.
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, trust_env=False) as client:
        res = await client.get(url)
        try:
            return res.status_code, res.json(), res.text
        except Exception:
            return res.status_code, None, res.text

def listify(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]

def summarize_display_payload(source: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data") or {}
    meta_json = data.get("meta_json") or {}
    weather = data.get("weather_info") or meta_json.get("weather_info") or {}
    image_url = data.get("image_url") or data.get("webp_url") or weather.get("image_url") or weather.get("webp_url") or weather.get("png_url") or ""
    webp_urls = listify(data.get("webp_urls") or weather.get("webp_urls") or meta_json.get("webp_urls") or [])
    png_urls = listify(data.get("png_urls") or weather.get("png_urls") or meta_json.get("png_urls") or [])
    image_urls = listify(data.get("image_urls") or weather.get("image_urls") or [])
    image_format = data.get("image_format") or weather.get("image_format")
    if not image_format:
        joined = " ".join(map(str, [image_url] + webp_urls + image_urls + png_urls)).lower()
        image_format = "webp" if ".webp" in joined else ("png" if ".png" in joined else "unknown")
    times = listify(data.get("times") or weather.get("times") or meta_json.get("times") or [])
    return {
        "source": source,
        "status": data.get("status", "unknown"),
        "message": data.get("message", ""),
        "business_type": data.get("business_type", source),
        "data_type": data.get("data_type", source),
        "source_file": data.get("source_file", ""),
        "source_files": data.get("source_files") or [],
        "meta_file": data.get("meta_file", ""),
        "image_format": image_format,
        "image_url": image_url,
        "webp_count": len(webp_urls),
        "png_count": len(png_urls),
        "image_count": len(image_urls),
        "time_count": len(times),
        "main_variable": weather.get("mainVariable") or weather.get("mainVariableName") or weather.get("变量") or "",
        "unit": weather.get("unit") or weather.get("单位") or "",
        "grid": weather.get("grid") or weather.get("网格") or "",
        "range": weather.get("range") or weather.get("范围") or "",
        "quality": weather.get("quality") or weather.get("status") or "",
        "min": weather.get("min"),
        "max": weather.get("max"),
        "mean": weather.get("mean"),
        "update": weather.get("update"),
    }

async def fetch_display(source: str) -> dict[str, Any]:
    source = normalize_source(source)
    url = f"{WEATHER_BACKEND}/api/display/{source}"
    status, payload, raw = await fetch_json(url)
    if status != 200:
        return {"ok": False, "url": url, "http_status": status, "payload": payload, "raw_text": raw[:1000], "summary": f"{source} 展示接口请求失败，HTTP 状态码为 {status}。"}
    if not isinstance(payload, dict) or payload.get("code") != 0:
        msg = payload.get("message") if isinstance(payload, dict) else raw[:200]
        return {"ok": False, "url": url, "http_status": status, "payload": payload, "raw_text": raw[:1000], "summary": f"{source} 展示接口返回异常：{msg}"}
    info = summarize_display_payload(source, payload)
    return {"ok": True, "url": url, "http_status": status, "payload": payload, "info": info, "summary": f"{source} 数据读取成功，当前主展示格式为 {info['image_format']}。"}

def wait_process_dir(source: str) -> Path:
    return DATA_DIR / normalize_source(source) / "wait_process"

def safe_count_files(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.rglob(pattern)) if root.exists() else 0

def latest_files(root: Path, patterns: list[str], limit: int = 10) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    files = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    files = sorted(set(files), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    out = []
    for p in files[:limit]:
        try:
            st = p.stat()
            out.append({"name": p.name, "path": str(p), "size": st.st_size, "modified": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")})
        except Exception:
            pass
    return out

def audit_assets(source: str) -> dict[str, Any]:
    source = normalize_source(source)
    root = wait_process_dir(source)
    counts = {
        "grib2": safe_count_files(root, "*.grib2"),
        "meta_json": safe_count_files(root, "*.meta.json"),
        "webp": safe_count_files(root, "*.webp"),
        "png": safe_count_files(root, "*.png"),
        "float32": safe_count_files(root, "*.float32"),
    }
    return {"source": source, "root": str(root), "exists": root.exists(), "counts": counts, "latest": latest_files(root, ["*.grib2", "*.meta.json", "*.webp", "*.png", "*.float32"], limit=12)}

async def run_subprocess(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(cwd), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, shell=False)
    chunks = []
    try:
        async with asyncio.timeout(MAX_SUBPROCESS_SECONDS):
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                chunks.append(line.decode("utf-8", errors="replace"))
            rc = await proc.wait()
            return rc, "".join(chunks)
    except TimeoutError:
        proc.kill()
        return -9, "".join(chunks) + f"\n[TIMEOUT] subprocess exceeded {MAX_SUBPROCESS_SECONDS}s\n"

def build_download_cmd(source: str, params: dict[str, Any]) -> list[str]:
    cmd = [PYTHON_BIN, str(DOWNLOAD_SCRIPT), "--source", normalize_source(source), "--lead-start", str(params["lead_start"]), "--lead-end", str(params["lead_end"]), "--lead-step", str(params["lead_step"]), "--min-success", "4", "--parse-after"]
    if params.get("overwrite"):
        cmd.append("--overwrite")
    if params.get("insecure_ssl"):
        cmd.append("--insecure-ssl")
    return cmd

async def handle_help() -> AsyncGenerator[str, None]:
    yield text_event("我是 GFS/ECMWF 气象数据智能体，可以查询数据、判断 WEBP、诊断不显示、审计资源、生成报告、触发下载解析。\n\n示例：\n- 检查 ECMWF 是否是 WEBP\n- 诊断 ECMWF 为什么不显示\n- 查询 GFS 最新数据\n- 审计 ECMWF 资源完整性\n- 生成 ECMWF 数据状态报告\n- 下载 ECMWF 到 72 小时\n- 下载 GFS 到 24 小时，覆盖并解析")
    yield done_event()

async def handle_check_format(text: str) -> AsyncGenerator[str, None]:
    source = normalize_source(text)
    yield text_event(f"正在检查 {source} 当前展示格式...\n")
    yield tool_event("get_display", f"请求 /api/display/{source}", 30)
    result = await fetch_display(source)
    if not result["ok"]:
        yield tool_event("get_display", "接口异常", 100, result["summary"], status="error")
        yield text_event(f"⚠️ {result['summary']}\n")
        yield done_event()
        return
    info = result["info"]
    yield tool_event("get_display", "展示接口正常", 100, f"image_format={info['image_format']}, webp_count={info['webp_count']}", status="done")
    is_webp = str(info["image_format"]).lower() == "webp" or ".webp" in str(info["image_url"]).lower()
    if is_webp:
        answer = f"{source} 当前主展示已经是 **WEBP**。\n\n- 主图像地址：`{info['image_url']}`\n- WEBP 时次数量：{info['webp_count']}\n- PNG 兜底数量：{info['png_count']}\n- 预报时次数量：{info['time_count']}\n\n结论：地图主图层走的是 WEBP，不是二进制。float32 如果存在，只是给点查或数值矩阵分析备用。"
    else:
        answer = f"{source} 当前主展示格式暂未判定为 WEBP。\n\n- 检测格式：`{info['image_format']}`\n- 主图像地址：`{info['image_url']}`\n- WEBP 数量：{info['webp_count']}\n- PNG 数量：{info['png_count']}"
    yield text_event(answer)
    yield done_event()

async def handle_query(text: str) -> AsyncGenerator[str, None]:
    source = normalize_source(text)
    yield text_event(f"正在查询 {source} 最新展示数据...\n")
    yield tool_event("query_display", f"查询 /api/display/{source}", 40)
    result = await fetch_display(source)
    if not result["ok"]:
        yield tool_event("query_display", "查询失败", 100, result["summary"], status="error")
        yield text_event(f"⚠️ {result['summary']}")
        yield done_event()
        return
    info = result["info"]
    yield tool_event("query_display", "查询完成", 100, "数据可用", status="done")
    answer = f"{source} 最新数据状态：\n\n- 状态：{info['status']}\n- 消息：{info['message']}\n- 主变量：{info['main_variable']}\n- 单位：{info['unit']}\n- 网格：{info['grid']}\n- 范围：{info['range']}\n- 主展示格式：{info['image_format']}\n- WEBP 时次：{info['webp_count']}\n- PNG 兜底：{info['png_count']}\n- 预报时次：{info['time_count']}\n- 更新时间：{info['update'] or '未提供'}"
    yield text_event(answer)
    yield done_event()

async def handle_diagnose(text: str) -> AsyncGenerator[str, None]:
    source = normalize_source(text)
    yield text_event(f"开始诊断 {source} 展示链路。\n")
    yield tool_event("display_api", f"检查 /api/display/{source}", 20)
    result = await fetch_display(source)
    if not result["ok"]:
        yield tool_event("display_api", "接口失败", 100, result["summary"], status="error")
        yield text_event(f"⚠️ {source} 后端展示接口异常。\n\n- 请求地址：`{result['url']}`\n- HTTP 状态：{result['http_status']}\n- 结论：先修 8002 主后端接口，再看前端图层。")
        yield done_event()
        return
    info = result["info"]
    audit = audit_assets(source)
    yield tool_event("display_api", "接口正常", 45, result["summary"], status="done")
    yield tool_event("asset_audit", "审计本地资源", 75, json.dumps(audit["counts"], ensure_ascii=False), status="done")
    yield tool_event("format_check", "检查 WEBP/PNG 格式", 100, f"format={info['image_format']}", status="done")
    answer = f"{source} 展示链路诊断结果：**基本正常**。\n\n## 1. 后端接口\n- 请求地址：`{result['url']}`\n- HTTP 状态：{result['http_status']}\n- 业务状态：{info['status']}\n\n## 2. 展示资源\n- 主展示格式：{info['image_format']}\n- 主图像 URL：`{info['image_url']}`\n- WEBP 数量：{info['webp_count']}\n- PNG 数量：{info['png_count']}\n- 时次数量：{info['time_count']}\n\n## 3. 本地资源审计\n- GRIB2：{audit['counts']['grib2']}\n- meta.json：{audit['counts']['meta_json']}\n- WEBP：{audit['counts']['webp']}\n- PNG：{audit['counts']['png']}\n- float32：{audit['counts']['float32']}\n\n如果前端仍不显示，下一步看浏览器 Network 中 `.webp` 请求是否 200，以及图层是否读取 `image_url` / `webp_urls`。"
    yield text_event(answer)
    yield done_event()

async def handle_audit(text: str) -> AsyncGenerator[str, None]:
    source = normalize_source(text)
    yield text_event(f"正在审计 {source} 本地资源完整性...\n")
    yield tool_event("audit_assets", f"扫描 {source}/wait_process", 30)
    audit = audit_assets(source)
    yield tool_event("audit_assets", "扫描完成", 100, json.dumps(audit["counts"], ensure_ascii=False), status="done")
    latest_text = "\n".join([f"- `{f['name']}` · {f['size']} bytes · {f['modified']}" for f in audit["latest"][:8]]) or "未发现文件。"
    answer = f"{source} 资源审计结果：\n\n- 目录：`{audit['root']}`\n- 目录存在：{audit['exists']}\n- GRIB2：{audit['counts']['grib2']}\n- meta.json：{audit['counts']['meta_json']}\n- WEBP：{audit['counts']['webp']}\n- PNG：{audit['counts']['png']}\n- float32：{audit['counts']['float32']}\n\n最近文件：\n{latest_text}"
    yield text_event(answer)
    yield done_event()

async def handle_report(text: str) -> AsyncGenerator[str, None]:
    source = normalize_source(text)
    yield text_event(f"正在生成 {source} 数据状态报告...\n")
    yield tool_event("collect_display", "收集展示接口状态", 30)
    display = await fetch_display(source)
    audit = audit_assets(source)
    if not display["ok"]:
        yield tool_event("collect_display", "接口异常", 100, display["summary"], status="error")
        yield text_event(f"⚠️ 报告生成失败：{display['summary']}")
        yield done_event()
        return
    info = display["info"]
    yield tool_event("collect_display", "展示状态收集完成", 60, status="done")
    yield tool_event("audit_assets", "资源审计完成", 80, json.dumps(audit["counts"], ensure_ascii=False), status="done")
    yield tool_event("generate_report", "生成报告文本", 100, status="done")
    report = f"# {source} 数值预报数据状态报告\n\n## 1. 数据状态\n- 数据源：{source}\n- 业务类型：{info['business_type']}\n- 数据类型：{info['data_type']}\n- 读取状态：{info['status']}\n- 质量标记：{info['quality'] or '未提供'}\n- 后端消息：{info['message']}\n\n## 2. 展示资源\n- 主展示格式：{info['image_format']}\n- 主图像 URL：`{info['image_url']}`\n- WEBP 资源数量：{info['webp_count']}\n- PNG 兜底数量：{info['png_count']}\n- 预报时次数量：{info['time_count']}\n\n## 3. 空间与变量信息\n- 主变量：{info['main_variable']}\n- 网格：{info['grid']}\n- 范围：{info['range']}\n- 单位：{info['unit']}\n- 最小值：{info['min']}\n- 最大值：{info['max']}\n- 平均值：{info['mean']}\n\n## 4. 本地文件审计\n- GRIB2：{audit['counts']['grib2']}\n- meta.json：{audit['counts']['meta_json']}\n- WEBP：{audit['counts']['webp']}\n- PNG：{audit['counts']['png']}\n- float32：{audit['counts']['float32']}\n\n## 5. 结论\n{source} 当前后端展示接口正常，前端可通过 `image_url` 或 `webp_urls` 加载图层。"
    yield text_event(report)
    yield done_event()

async def handle_download(text: str) -> AsyncGenerator[str, None]:
    source = normalize_source(text)
    params = parse_lead_params(text, source)
    if not DOWNLOAD_SCRIPT.exists():
        yield text_event(f"⚠️ 找不到下载脚本：`{DOWNLOAD_SCRIPT}`。\n\n请检查环境变量 `BACKEND_SYSTEM_DIR` 是否指向 backend_system 目录。")
        yield done_event()
        return
    cmd = build_download_cmd(source, params)
    yield text_event(f"准备下载并解析 {source} 数据。\n\n- lead_start：{params['lead_start']}\n- lead_end：{params['lead_end']}\n- lead_step：{params['lead_step']}\n- overwrite：{params['overwrite']}\n- insecure_ssl：{params['insecure_ssl']}\n\n开始执行后端下载解析脚本...\n")
    yield tool_event("download_parse", "启动下载解析脚本", 10, " ".join(cmd))
    rc, output = await run_subprocess(cmd, cwd=BACKEND_SYSTEM_DIR)
    tail = output[-3000:] if output else ""
    if rc == 0:
        yield tool_event("download_parse", "下载解析完成", 100, tail, status="done")
        yield text_event(f"{source} 下载解析任务执行完成。\n\n退出码：{rc}\n\n关键日志：\n```text\n{tail}\n```\n\n你可以继续问：检查 ECMWF 是否是 WEBP / 审计 ECMWF 资源完整性。")
    else:
        yield tool_event("download_parse", "下载解析失败", 100, tail, status="error")
        yield text_event(f"⚠️ {source} 下载解析失败。\n\n退出码：{rc}\n\n日志尾部：\n```text\n{tail}\n```")
    yield done_event()



async def handle_compare_sources(text: str) -> AsyncGenerator[str, None]:
    sources = ["GFS", "ECMWF"]

    yield text_event("正在同时检查 GFS 和 ECMWF 数据状态...\n")
    yield tool_event("compare_sources", "启动双源状态检查", 10, "GFS + ECMWF")

    rows = []
    for i, source in enumerate(sources, start=1):
        yield tool_event("compare_sources", f"检查 {source}", 10 + i * 30)

        display = await fetch_display(source)
        audit = audit_assets(source)

        if display.get("ok"):
            info = display.get("info") or {}
            rows.append({
                "source": source,
                "ok": True,
                "status": info.get("status"),
                "format": info.get("image_format"),
                "webp": info.get("webp_count"),
                "png": info.get("png_count"),
                "times": info.get("time_count"),
                "image_url": info.get("image_url"),
                "grib2": audit["counts"]["grib2"],
                "meta_json": audit["counts"]["meta_json"],
                "float32": audit["counts"]["float32"],
            })
        else:
            rows.append({
                "source": source,
                "ok": False,
                "status": "error",
                "format": "unknown",
                "webp": 0,
                "png": 0,
                "times": 0,
                "image_url": "",
                "grib2": audit["counts"]["grib2"],
                "meta_json": audit["counts"]["meta_json"],
                "float32": audit["counts"]["float32"],
                "error": display.get("summary"),
            })

    yield tool_event("compare_sources", "双源检查完成", 100, "GFS + ECMWF", status="done")

    lines = []
    lines.append("| 数据源 | 接口 | 主展示 | WEBP | PNG | 时次 | GRIB2 | meta.json | float32 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        ok_text = "正常" if r["ok"] else "异常"
        lines.append(
            f"| {r['source']} | {ok_text} | {r['format']} | {r['webp']} | {r['png']} | {r['times']} | {r['grib2']} | {r['meta_json']} | {r['float32']} |"
        )

    summary = "\n".join(lines)

    all_ok = all(r["ok"] for r in rows)
    all_webp = all(str(r["format"]).lower() == "webp" for r in rows if r["ok"])

    conclusion = []
    if all_ok:
        conclusion.append("GFS 和 ECMWF 展示接口均正常。")
    else:
        conclusion.append("至少一个数据源展示接口异常，需要检查 8002 主后端。")

    if all_webp:
        conclusion.append("两个数据源当前主展示均为 WEBP。")
    else:
        conclusion.append("至少一个数据源主展示格式不是 WEBP 或未能识别。")

    yield text_event(
        "## GFS / ECMWF 双源状态对比\n\n"
        + summary
        + "\n\n## 结论\n"
        + "\n".join(f"- {x}" for x in conclusion)
    )

    yield done_event()


async def handle_chart(text: str) -> AsyncGenerator[str, None]:
    source = normalize_source(text)
    yield text_event(f"正在生成 {source} 当前预报图层展示...\n")
    yield tool_event("collect_display", f"读取 /api/display/{source}", 30)

    result = await fetch_display(source)

    if not result["ok"]:
        yield tool_event("collect_display", "接口异常", 100, result["summary"], status="error")
        yield text_event(f"⚠️ 无法生成图表：{result['summary']}")
        yield done_event()
        return

    info = result["info"]
    raw_url = info.get("image_url") or ""
    image_url = make_backend_asset_url(raw_url)

    yield tool_event(
        "collect_display",
        "图层资源读取完成",
        70,
        f"format={info['image_format']}, url={raw_url}",
        status="done",
    )

    if not image_url:
        yield tool_event("render_image", "图像地址缺失", 100, "后端未返回 image_url/webp_url", status="error")
        yield text_event(
            f"⚠️ {source} 后端没有返回可展示图像地址。\n\n"
            f"请检查 `/api/display/{source}` 返回中是否包含 `image_url` 或 `webp_url`。"
        )
        yield done_event()
        return

    yield tool_event("render_image", "生成图表完成", 100, image_url, status="done")

    yield text_event(
        f"{source} 当前图层已生成。\n\n"
        f"- 展示格式：{info['image_format']}\n"
        f"- WEBP 数量：{info['webp_count']}\n"
        f"- PNG 兜底数量：{info['png_count']}\n"
        f"- 图像地址：`{image_url}`\n"
    )

    yield image_event(image_url, f"{source} 当前预报图层")
    yield done_event()


async def handle_call_model(text: str) -> AsyncGenerator[str, None]:
    source = normalize_source(text)

    yield text_event(
        f"已收到 {source} 模型调用请求。\n\n"
        "当前版本已经接入数据查询、显示诊断、资源审计、图表生成和下载解析。\n"
        "真实模型推理接口还没有在 8002 主后端注册，所以我暂时不会伪造模型结果。\n"
    )

    yield tool_event("model_registry", "检查模型调用能力", 60, "pending backend model API")

    yield text_event(
        "下一步需要在 8002 主后端补一个模型接口，例如：\n\n"
        "`POST /api/model/run`\n\n"
        "建议请求参数：\n"
        "- source: GFS / ECMWF\n"
        "- variable: t2m / tp / sp / d2m\n"
        "- lead_time: 0 / 3 / 6 / ...\n"
        "- model_name: forecast_correction / risk_warning / nowcast\n\n"
        "等这个接口有了，我就可以把 `/调用模型` 真正接成自动工具调用。"
    )

    yield tool_event("model_registry", "模型接口尚未接入", 100, "需要后端新增 /api/model/run", status="done")
    yield done_event()


async def handle_delete_guard(text: str) -> AsyncGenerator[str, None]:
    yield text_event("删除/清理属于高风险操作，当前智能体不会直接自动执行。\n\n建议流程：\n1. 先说：`审计 ECMWF 资源完整性`\n2. 确认要删除的数据集和目录\n3. 后续版本可加入确认码，例如 `DELETE_ECMWF_20260702`\n\n当前版本只提供诊断、查询、下载解析和报告生成，不执行自动删除。")
    yield done_event()

async def handle_chat(text: str) -> AsyncGenerator[str, None]:
    yield text_event("我已经接入 GFS/ECMWF 数据诊断与运维工具。你可以直接说：\n\n- 检查 ECMWF 是否是 WEBP\n- 查询 GFS 最新数据\n- 诊断 ECMWF 为什么不显示\n- 审计 ECMWF 资源完整性\n- 生成 ECMWF 数据状态报告\n- 下载 ECMWF 到 72 小时")
    yield done_event()

@app.get("/api/agent/health")
def health() -> dict[str, Any]:
    return {
        "code": 0,
        "message": "agent online",
        "weather_backend": WEATHER_BACKEND,
        "backend_system_dir": str(BACKEND_SYSTEM_DIR),
        "download_script_exists": DOWNLOAD_SCRIPT.exists(),
        "data_dir": str(DATA_DIR),
    }

@app.get("/api/agent/tools")
def tools() -> dict[str, Any]:
    return {"code": 0, "data": [
        {"name": "query", "description": "查询 GFS/ECMWF 最新展示状态", "examples": ["查询 ECMWF 最新数据", "查询 GFS 状态"]},
        {"name": "check_format", "description": "判断当前展示是否为 WEBP", "examples": ["检查 ECMWF 是否是 WEBP", "现在 GFS 是二进制还是 WEBP"]},
        {"name": "diagnose", "description": "诊断前后端展示链路", "examples": ["诊断 ECMWF 为什么不显示"]},
        {"name": "audit", "description": "审计本地资源完整性", "examples": ["审计 ECMWF 资源完整性"]},
        {"name": "report", "description": "生成状态报告", "examples": ["生成 ECMWF 数据状态报告"]},
        {"name": "download", "description": "下载并解析 GFS/ECMWF", "examples": ["下载 ECMWF 到 72 小时", "下载 GFS 到 24 小时，覆盖并解析"]},
    ]}

@app.post("/api/agent/chat")
async def chat(req: AgentChatRequest):
    text = last_user_text(req.messages)

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            intent = detect_intent(text)
            if intent == "help":
                async for item in handle_help(): yield item
            elif intent == "download":
                async for item in handle_download(text): yield item
            elif intent == "audit":
                async for item in handle_audit(text): yield item
            elif intent == "diagnose":
                async for item in handle_diagnose(text): yield item
            elif intent == "check_format":
                async for item in handle_check_format(text): yield item
            elif intent == "compare_sources":
                async for item in handle_compare_sources(text):
                    yield item
            elif intent == "chart":
                async for item in handle_chart(text):
                    yield item
            elif intent == "call_model":
                async for item in handle_call_model(text):
                    yield item
            elif intent == "report":
                async for item in handle_report(text): yield item
            elif intent == "query":
                async for item in handle_query(text): yield item
            elif intent == "delete_guard":
                async for item in handle_delete_guard(text): yield item
            else:
                async for item in handle_chat(text): yield item
        except Exception as e:
            yield error_event(str(e))
            yield done_event()

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
