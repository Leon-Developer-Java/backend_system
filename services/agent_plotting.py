"""用 matplotlib 生成 PNG 折线图，落到 outputs/，返回可访问 URL。"""
from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")  # 无界面后端，服务器环境必备
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

_BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = _BASE_DIR / "data" / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_BASE_URL = os.getenv(
    "BACKEND_SYSTEM_PUBLIC_BASE_URL",
    "http://127.0.0.1:8002",
).rstrip("/")

# 中文字体回退：找到任意可用中文字体则用，否则标题降级英文避免乱码
_CN_FONTS = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "PingFang SC"]
_available = {f.name for f in font_manager.fontManager.ttflist}
_CN_FONT = next((name for name in _CN_FONTS if name in _available), None)
if _CN_FONT:
    rcParams["font.sans-serif"] = [_CN_FONT]
    rcParams["axes.unicode_minus"] = False

HAS_CN_FONT = _CN_FONT is not None

# 默认示例时序（8 个时次），当调用方不提供数据时使用
DEFAULT_TIMES = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]
DEFAULT_SERIES = {
    "气温(℃)": [21, 22, 24, 27, 30, 29, 26, 23],
    "降水(mm)": [0, 0.2, 0.8, 1.6, 2.4, 1.2, 0.4, 0.1],
}


def line_chart(title: str, series: dict[str, list[float]] | None = None,
               x: list | None = None) -> str:
    """画折线图并保存为 PNG，返回对外可访问的 URL。

    series: {系列名: [数值,...]}；缺省用内置示例数据。
    """
    series = series or DEFAULT_SERIES
    x = x or DEFAULT_TIMES
    safe_title = title if HAS_CN_FONT else "Weather Chart"

    fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
    for name, values in series.items():
        label = name if HAS_CN_FONT else _ascii_fallback(name)
        n = min(len(x), len(values))
        ax.plot(x[:n], values[:n], marker="o", linewidth=2, label=label)
    ax.set_title(safe_title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    name = f"{uuid4().hex}.png"
    fig.savefig(OUTPUTS_DIR / name)
    plt.close(fig)
    return f"{PUBLIC_BASE_URL}/data/outputs/{name}"


def _ascii_fallback(name: str) -> str:
    """无中文字体时给系列名一个英文回退标签。"""
    table = {"气温(℃)": "Temp", "降水(mm)": "Precip", "气压(hPa)": "Pressure", "风速(m/s)": "Wind"}
    return table.get(name, "series")
