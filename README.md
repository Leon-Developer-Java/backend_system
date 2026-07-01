# 气象数据展示后台

FastAPI 后台。当前版本只做一件事：接收前端上传的气象文件，按业务类型保存到 `data/`，调用对应 adapter 生成 `meta.json`，并把展示面板需要的数据返回给前端。

Himawari 当前已经接入自动下载、HSD 解析、等经纬度 PNG/meta 生成和前端展示窗口补齐逻辑。其他业务仍按各自 adapter/service 维护。

## 目录结构

```text
backend/
├─ main.py                  # FastAPI 入口
├─ adapters/                # 6 个业务处理脚本
│  ├─ base.py
│  ├─ cma_adapter.py
│  ├─ era5_adapter.py
│  ├─ gfs_adapter.py
│  ├─ himawari_adapter.py
│  ├─ radar_adapter.py
│  └─ wrf_adapter.py
├─ data/                    # 上传文件、meta.json、后续 PNG 都放这里
│  ├─ CMA/
│  ├─ ERA5/
│  ├─ GFS/
│  ├─ Himawari/
│  ├─ Radar/
│  └─ WRF/
├─ services/                # 前端按数据类型读取展示数据
│  ├─ cma_service.py
│  ├─ era5_service.py
│  ├─ gfs_service.py
│  ├─ himawari_service.py
│  ├─ radar_service.py
│  └─ wrf_service.py
├─ samples/                 # 小样例文件
├─ requirements.txt
└─ README.md
```

## 当前接口

```text
GET  /api/health
POST /api/files/parse
GET  /api/display/{business_type}
GET  /api/himawari/auto-status
```

前端当前调用：

```text
POST http://127.0.0.1:8002/api/files/parse
```

上传字段名必须是 `file`。

前端点击数据类型时可调用：

```text
GET http://127.0.0.1:8002/api/display/ERA5
```

返回对应业务目录下最新的 `meta_json` 和 PNG 路径。

Himawari 前端轮询：

```text
GET http://127.0.0.1:8002/api/display/HIMAWARI
GET http://127.0.0.1:8002/api/himawari/auto-status
```

`/api/display/HIMAWARI` 返回当前滚动窗口内已有的 `timeline`、`variables`、`composites`、`png_url`、`extent` 和 `grid`。窗口规则为：

```text
窗口右边界 = 当前时间 - 60 分钟后按 10 分钟取整
窗口左边界 = 窗口右边界 - 24 小时
```

窗口内缺失时次会由自动下载逐轮补齐；未补齐前不会在前端时间轴显示。

## 业务识别规则

后端会根据文件名或扩展名判断业务类型：

```text
CMA       -> 文件名包含 cma
ERA5      -> 文件名包含 era5，或 .nc 默认归入 ERA5
GFS       -> 文件名包含 gfs，或 .grib/.grib2
Himawari  -> 文件名包含 himawari/hsd，或 .hsd
Radar     -> 文件名包含 radar/cinrad，或 .cinrad/.radar
WRF       -> 文件名包含 wrf
```

如果 `.nc` 文件属于 WRF，文件名中需要包含 `wrf`。

## Adapter 规则

每个成员只改自己负责的 adapter：

```text
CMA       -> adapters/cma_adapter.py
ERA5      -> adapters/era5_adapter.py
GFS       -> adapters/gfs_adapter.py
Himawari  -> adapters/himawari_adapter.py
Radar     -> adapters/radar_adapter.py
WRF       -> adapters/wrf_adapter.py
```

每个 adapter 对外保留这个函数：

```python
def process_file(file_path: str, data_type: str) -> dict:
    ...
```

Himawari adapter 已实现 HSD raw 分段下载、断点续传、等经纬度网格重采样、PNG/meta 输出和解析成功后 raw 清理。其他 adapter 按各自业务继续维护。

## Himawari 自动下载与解析

Himawari 自动下载需要通过环境变量传入 FTP 凭据，账号密码不要写入源码、README 或提交信息：

```bash
export HIMAWARI_FTP_USER="你的 FTP 用户名"
export HIMAWARI_FTP_PASSWORD="你的 FTP 密码"
```

常用配置：

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `HIMAWARI_AUTO_DOWNLOAD` | `1` | 后端启动后自动下载；设为 `0/false/no/off` 可关闭 |
| `HIMAWARI_WINDOW_HOURS` | `24` | 展示窗口和补齐窗口长度 |
| `HIMAWARI_LATEST_DELAY_MINUTES` | `60` | 当前时间向前延迟后作为窗口右边界 |
| `HIMAWARI_DOWNLOAD_INTERVAL_MINUTES` | `10` | 下载时次间隔 |
| `HIMAWARI_DOWNLOAD_INTERVAL_SECONDS` | `60` | 自动任务轮询间隔 |
| `HIMAWARI_DOWNLOAD_MAX_JOBS_PER_RUN` | `12` | 每轮最多处理的时次/通道任务数；`0` 表示不限量 |
| `HIMAWARI_FILE_WORKERS` | `4` | 单个时次 HSD 分段文件并发下载上限 |
| `HIMAWARI_BANDS` | `B13,B03,B02,B01` | 当前只建议使用这四个通道 |

当前策略：

- 自动窗口右边界为“当前时间 - 60 分钟”的 10 分钟整点，向前覆盖 24 小时。
- 优先补齐最新缺失时次的 `B13` 红外窗口亮温。
- 白天时次再补 `B03/B02/B01`，用于生成真彩色云图。
- 不再自动下载 `B04-B16` 全通道。
- 下载失败时保留较新的 `.part/raw` 供下一轮断点续传；解析成功后删除对应 raw。
- 过期解析结果按滚动 24 小时窗口清理。

HSD 原始数据不提交到 Git；需要别人拉取后看到效果时，只提交必要的解析结果 `meta.json + PNG`，不要提交 raw、`.part`、`.float32` 或 `.nc` 中间文件。

## 启动

推荐使用项目当前 conda 环境：

```bash
cd zhihuiqixiangSQL/backend_system
conda activate zhihuiqixiang
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8002
```

只验证已有 Himawari 展示结果、不希望启动后自动下载时：

```bash
cd zhihuiqixiangSQL/backend_system
conda activate zhihuiqixiang
export HIMAWARI_AUTO_DOWNLOAD=0
uvicorn main:app --reload --host 127.0.0.1 --port 8002
```

需要自动下载 Himawari 时：

```bash
cd zhihuiqixiangSQL/backend_system
conda activate zhihuiqixiang
export HIMAWARI_FTP_USER="你的 FTP 用户名"
export HIMAWARI_FTP_PASSWORD="你的 FTP 密码"
uvicorn main:app --reload --host 127.0.0.1 --port 8002
```

访问：

```text
http://127.0.0.1:8002/docs
```

## 返回给前端的数据

`/api/files/parse` 返回统一格式：

```json
{
  "code": 0,
  "data": {
    "file_name": "era5_sample.nc",
    "directory": "D:/weather_prediction_system/backend/data/ERA5/",
    "business_type": "ERA5",
    "meta": {},
    "weather_info": {}
  },
  "message": "success"
}
```

前端当前使用：

```text
data.file_name
data.directory
data.weather_info
```

## 协作注意

- 不提交大体积气象数据。
- 不提交 Himawari HSD raw、`.part`、`.float32`、`.nc` 中间文件。
- Himawari 解析后的展示样例可以提交 `meta.json + PNG`，用于前端默认展示。
- 小样例放 `samples/`。
- 上传或处理后的文件按业务放入 `data/{业务名}/`。
- 公共字段不够用时，先放到 meta 的 `extra` 中。
- 新增 Python 依赖时必须同步更新 `requirements.txt`。
