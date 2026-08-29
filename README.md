# 气象展示与 Adapter 后端（backend，端口 8002）

`backend` 是气象展示 API、静态产物服务和 Adapter Worker 的代码仓库。前端从这里读取 `/api/display/...` 与 `/data/...`；正式上传由 `backend_upload:8003` 接收，解析任务通过共享数据库交给 Worker 异步执行。

## 运行环境

- Python 3.12（科学计算依赖的当前验证版本）
- 工作区必须同时包含同级的 `DB/`、`storage/` 和 `backend_upload/`
- `workers/` 是本仓库源码包，不是 pip 包；克隆或合并时必须完整保留 `workers/__init__.py`、`launcher.py`、`parse_worker.py` 和 `adapter_subprocess.py`

首次安装：

```powershell
cd D:\weather_prediction_system\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

`cfgrib/eccodes`、`rasterio`、`h5py`、`netCDF4`、`satpy` 等包体积较大。安装完成后建议执行 `python -m pip check`。ERA5 CDS 下载还需要在 `%USERPROFILE%\.cdsapirc` 中配置 CDS API 凭据。

## 启动

开发模式：

```powershell
cd D:\weather_prediction_system\backend
.\.venv\Scripts\python.exe main.py
```

服务地址：

- API：`http://127.0.0.1:8002`
- OpenAPI：`http://127.0.0.1:8002/docs`
- 健康检查：`http://127.0.0.1:8002/api/health`

默认 `START_ADAPTER_WORKER_WITH_API=true`，FastAPI 生命周期会启动一个 `python -m workers.parse_worker` 子进程。生产环境运行多个 API 进程时必须关闭该选项，并单独托管一个 Worker：

```powershell
$env:START_ADAPTER_WORKER_WITH_API = "false"
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8002

# 另一个终端/服务
.\.venv\Scripts\python.exe -m workers.parse_worker
```

SQLite 模式下 Worker 使用 `storage/tmp/adapter/parse_worker.lock` 保证单实例。不要同时启动 API 内置 Worker 和独立 Worker。

## 当前架构

```text
backend_upload :8003
  -> raw 文件写入 storage/raw
  -> public_info / satellite_collection 入队

Adapter Worker
  -> 从共享 DB 领取 pending/ready 任务
  -> 每次启动一个 adapter_subprocess
  -> 在临时目录生成并校验 meta/WebP
  -> 发布到 backend/data/{data_type}/assets/{file_uuid}
  -> 事务写入类型明细表并更新任务状态

backend :8002
  -> 返回展示 meta、时次、变量、范围和 WebP URL
  -> 通过 /data 提供已发布产物
```

正式链路支持 ERA5、GFS、ECMWF、CMA、Radar、WRF、FY-3 和 Himawari。FY-3/Himawari 集合由 Worker 作为一个工作单元解析；集合不完整时不会进入解析队列。

`POST /api/files/parse`、`POST /api/files/raw-upload` 以及旧卫星 raw 场景接口仍用于兼容和调试。前端正式上传页不应把文件再次提交到这些接口。

## 共享数据库与目录

默认位置：

```text
D:\weather_prediction_system\DB\weather.db       # 共享 SQLite，运行时生成
D:\weather_prediction_system\storage\raw         # 私有原始文件
D:\weather_prediction_system\storage\tmp         # 上传/Adapter 临时文件
D:\weather_prediction_system\backend\data         # 可展示 meta/WebP
```

`backend_auth`、`backend_upload` 和 Worker 必须使用同一个 `DATABASE_URL`。`DB/weather.db` 不提交 Git；启动服务时 `DB.migrate.init_database()` 会创建或升级表结构。切勿只复制 `backend/` 目录而遗漏工作区根目录的 `DB` Python 包。

常用环境变量：

| 变量 | 默认值/用途 |
| --- | --- |
| `DATABASE_URL` | `sqlite:///D:/weather_prediction_system/DB/weather.db` |
| `JWT_SECRET` | 必须与 auth/upload/agent/model 一致 |
| `RAW_STORAGE_ROOT` | 工作区 `storage/raw` |
| `TMP_STORAGE_ROOT` | 工作区 `storage/tmp` |
| `PRODUCT_DATA_ROOT` | `backend/data` |
| `START_ADAPTER_WORKER_WITH_API` | 是否随 8002 启动 Worker，默认 `true` |
| `ADAPTER_QUEUE_SLEEP_SECONDS` | 空队列轮询间隔，默认 2 秒 |
| `ADAPTER_LEASE_SECONDS` | 任务租约，默认 1800 秒 |
| `ADAPTER_HEARTBEAT_SECONDS` | Worker 续租间隔，默认 60 秒 |
| `ADAPTER_MAX_ATTEMPTS` | 最大解析次数，默认 3 |
| `ADAPTER_RETRY_BACKOFF_SECONDS` | 默认 `60,300,1800` |
| `ADAPTER_TIMEOUT_SECONDS` | Adapter 默认超时，默认 3600 秒 |
| `CORS_ORIGINS` | 允许访问 8002 的前端来源 |
| `ENABLE_LEGACY_HIMAWARI_SCHEDULER` | 旧调度器开关，保持 `false` |

自动采集由 `backend_upload` 负责；不要同时开启这里的旧 Himawari 调度器。

## 主要接口

```text
GET  /api/health
GET  /api/display/{business_type}
GET  /api/ERA5/datasets
GET  /api/ERA5/datasets/{dataset_id}
GET  /api/ERA5/datasets/{dataset_id}/assets
POST /api/wrf/rescan
POST /api/files/parse                         # 兼容/调试
POST /api/files/raw-upload                    # 兼容/调试
GET  /api/display/{business_type}/raw-scenes  # 旧卫星兼容
POST /api/display/{business_type}/parse-tasks # 旧卫星兼容
```

展示响应继续兼容 `meta_json`、`webp`、`webp_files`、`variable_layers`、`resolution_options`、`times`、`extent` 和 `weather_info`。`/data` 只应包含可发布产物，原始文件不得放入或暴露在该目录。

## 代码结构

```text
backend/
├─ main.py
├─ auth.py
├─ adapters/                  # 各数据类型解析器
├─ services/                  # 展示、产物目录和 Adapter 调度
├─ workers/
│  ├─ launcher.py             # API 生命周期启动/停止 Worker
│  ├─ parse_worker.py         # DB 队列、租约、重试与提交
│  └─ adapter_subprocess.py   # 单任务隔离进程入口
├─ scripts/                   # ERA5、GFS/ECMWF 下载工具
├─ data/                      # 展示产物
└─ requirements.txt
```

## 验证

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m py_compile main.py workers\launcher.py workers\parse_worker.py
```

修改 Adapter 时至少使用对应数据类型的小样例验证 meta、WebP、`extent`、时次顺序和数据库明细路径。不要提交 `.env`、共享数据库、raw 数据或临时解析目录。
