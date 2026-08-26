# 气象数据展示与解析后台

`backend` 是现有 FastAPI 展示后端，负责气象数据展示接口、Adapter 代码和解析产物访问，默认端口为 8002。

数据库和上传功能接入后，原始文件由 `backend_upload` 接收并登记到 `public_info`；本后端已新增独立的 Adapter Worker 进程，从数据库领取待解析任务，调用 Adapter 生成 meta、WebP 和数据类型明细记录。

当前已实现共享数据库队列、单实例单并发 Worker、Adapter 子进程、租约与重试、明细入库和已发布子目录展示兼容。FY-3/Himawari 多文件上传通过集合队列自动判断完整性并由同一 Worker 解析；自动下载仍不在本版本范围。

## 本版本决策

- 不参考当前 `services/himawari_scheduler.py` 的定时任务实现。它是之前的测试代码，不属于本版本架构设计或验收范围。
- 自动下载功能暂不实现。未来下载器只需保存 raw 并写入 `public_info(parse_status=pending)`，后续解析复用相同 Worker。
- 不引入 Redis、Celery 或其他外部消息队列。
- `public_info` 直接作为持久化解析队列和任务状态的唯一事实来源。
- `main.py` 默认随 FastAPI 生命周期启动一个独立 Adapter Worker 子进程，但不新增 HTTP 端口或前端直连入口。
- 第一版 Worker 为单实例、单并发：同一时间只执行一个 Adapter 任务。
- Adapter 使用独立子进程执行；Worker 主进程等待子进程结束，不并行启动第二个 Adapter。
- 上传请求内不运行 Adapter，前端也不再把同一文件二次提交给 `/api/files/parse`。

## 服务边界

```text
前端
  -> backend_upload :8003
       -> 分片上传、合并、SHA-256、raw 落盘
       -> public_info.ingest_status=success
       -> public_info.parse_status=pending

Adapter Worker（无 HTTP 端口）
  -> 优先领取 satellite_collection.ready，再领取普通 public_info.pending
  -> 启动一个 Adapter 子进程
  -> 生成 meta/WebP
  -> 写数据类型明细表
  -> 集合全部成员或普通任务的 parse_status=success/failed

backend :8002
  -> /api/display/...
  -> /data/... WebP
  -> 保持现有前端展示响应兼容
```

各模块职责：

| 模块 | 负责 | 不负责 |
| --- | --- | --- |
| `backend_upload` | 上传、校验、raw 文件、`public_info` 入队 | 调用 Adapter、生成 WebP |
| Adapter Worker | 任务领取、子进程管理、租约、重试、结果入库 | 提供 HTTP 接口、接收上传 |
| Adapter 子进程 | 读取一个工作单元、执行气象解析、生成临时产物 | 领取任务、直接更新任务状态 |
| `backend` FastAPI | 展示接口、静态 WebP、资源查询 | 承载长时间 Adapter 任务 |
Himawari 当前已经接入自动下载、HSD 解析、等经纬度 WebP/meta 生成和前端展示窗口补齐逻辑。其他业务仍按各自 adapter/service 维护。

## 目录结构

当前主要目录：

```text
backend/
├─ main.py
├─ adapters/
│  ├─ base.py
│  ├─ cma_adapter.py
│  ├─ era5_adapter.py
│  ├─ fy3_adapter.py
│  ├─ gfs_adapter.py
│  ├─ himawari_adapter.py       # 当前代码保留，但不属于本版定时任务方案
│  ├─ radar_adapter.py
│  └─ wrf_adapter.py
├─ services/
├─ data/                        # 现有 /data 展示产物根目录
├─ samples/
├─ data/                    # 上传文件、meta.json、后续 PNG 都放这里
│  ├─ CMA/
│  ├─ ERA5/
│  ├─ GFS/
│  ├─ FY3/
│  │  ├─ raw/              # 所有 FY-3 HDF 原始文件，不按日期分目录
│  │  └─ YYYYMMDD/HHMM/    # 仅存解析后的 WebP 和 scene.meta.json
│  ├─ Himawari/
│  │  ├─ raw/              # 所有 HSD 原始文件和续传 part，不按日期分目录
│  │  └─ YYYYMMDD/HHMM/    # 仅存解析后的 WebP 和 scene.meta.json
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

当前已新增：

```text
backend/
├─ services/
│  └─ adapter_runner.py         # Adapter 分发、结果标准化和产物校验
└─ workers/
   ├─ parse_worker.py           # 单实例任务循环、租约、重试、数据库提交
   └─ adapter_subprocess.py      # 单个任务的 Adapter 子进程入口
```

原始文件和解析产物使用不同根目录：

```text
D:\weather_prediction_system\storage\raw\     # 私有原始文件，不挂载为 /data
D:\weather_prediction_system\storage\tmp\     # 上传和 Adapter 临时文件
D:\weather_prediction_system\backend\data\    # meta/WebP 展示产物
```

Worker 只根据 `public_info.source_path` 读取 raw，不扫描 `wait_process` 或数据类型目录发现任务。

## 当前接口

当前主后端仍提供：

```text
GET  /api/health
POST /api/files/parse
POST /api/files/raw-upload
GET  /api/display/{business_type}
```

`POST /api/files/parse` 当前会接收文件并同步调用 Adapter。数据库队列完成后，该接口只作为过渡兼容或调试入口，正式上传链路不再调用它。

现有展示调用保持：

```text
GET http://127.0.0.1:8002/api/display/ERA5
GET http://127.0.0.1:8002/api/display/GFS
GET http://127.0.0.1:8002/api/display/WRF
```

数据库队列接入后仍保持现有展示响应中的以下字段：

```text
meta_json
webp
webp_files
variable_layers
resolution_options
times
extent
weather_info
```

## public_info 任务队列

上传完成后至少写入：

```text
ingest_status      = success
parse_status       = pending
parse_attempts     = 0
next_parse_at      = now
source_path        = raw 相对路径
data_type          = 标准化数据类型
is_deleted         = false
```

Worker 领取条件：

```text
ingest_status = success
parse_status = pending
next_parse_at <= now
is_deleted = false
```

数据库支持时使用 `SELECT ... FOR UPDATE SKIP LOCKED`，或使用等价的条件更新，保证同一任务只能被一个 Worker 从 `pending` 改为 `running`。

领取成功后写入：

```text
parse_status       = running
parse_worker       = {hostname}:{pid}:{instance_uuid}
parse_started_at   = now
parse_lease_until  = now + lease
parse_attempts     = parse_attempts + 1
parse_error        = null
```

任务状态：

```text
pending -> running -> success
                   -> pending   # 可重试失败
                   -> failed    # 超过重试或不可恢复错误
```

## Worker 执行模型

### 单实例与单并发

第一版只允许启动一个 Worker 实例。默认开发库是 SQLite，因此当前使用 `storage/tmp/adapter/parse_worker.lock` 的操作系统排他锁；进程异常退出时锁由操作系统释放。切换 MySQL/PostgreSQL 后可把这一层替换为数据库 advisory lock，任务本身仍由数据库条件更新和租约保证不重复领取。

正常情况下最多存在一个由当前 Worker 执行的 `running` 任务，但 Worker 不能通过“数据库中存在任意 running 就停止”来控制并发。旧 Worker 崩溃后可能遗留 `running`，必须依靠任务租约恢复。

Worker 主循环：

```python
while True:
    task = claim_one_pending_task()

    if task is None:
        sleep(2)
        continue

    mark_running(task)

    try:
        result = run_one_adapter_subprocess(task)
        validate_and_commit(result)
        mark_success(task)
    except RetryableError as error:
        schedule_retry(task, error)
    except Exception as error:
        mark_failed(task, error)
```

`sleep(2)` 只在没有可领取任务时执行。Adapter 子进程运行期间，主 Worker 等待当前任务结束，不领取下一条任务，也不启动第二个 Adapter。

多个文件同时进入队列时按以下稳定顺序串行处理：

```text
ORDER BY next_parse_at ASC, create_time ASC, id ASC
```

### Adapter 子进程

每个任务启动一个独立子进程，而不是在 Worker 主进程中直接解析：

```text
Worker 主进程
  -> 创建 storage/tmp/adapter/{file_uuid}/{attempt_id} 保存 job/result/error/log
  -> 创建 backend/data/{data_type}/.adapter_staging/{file_uuid}/{attempt_id} 暂存 Adapter 产物
  -> 写入只包含任务参数的 job.json
  -> 启动 adapter_subprocess
  -> 等待子进程并定期续租
  -> 读取 result.json / error.json
  -> 校验产物
  -> 主进程提交数据库
  -> 清理临时目录
```

同一时间只允许存在一个由 Worker 启动的 Adapter 子进程。这是进程隔离，不是多线程并发。

子进程的职责：

- 读取 `source_path`、`data_type`、`output_root`、`file_uuid` 和可选 `collection_uuid`。
- 集合任务读取全部成员 `source_path`，复制到独立临时输入目录后再调用多文件 Adapter。
- 调用固定白名单中的 Adapter。
- 将 meta、WebP 和中间产物先写入 `.adapter_staging`；发布前该目录不参与展示服务的 meta 扫描。
- 输出标准化 `result.json`，不直接修改 `public_info`。
- 通过退出码和 `error.json` 返回失败，不静默生成“成功”占位结果。

主 Worker 的职责：

- 管理超时、终止异常子进程并记录退出码。
- Adapter 运行期间定期续租，避免长任务被错误回收。
- 校验 meta、WebP 数量、路径、文件大小和 `assets[]`。
- 将验证后的产物原子移动到正式展示目录。
- 在数据库事务中写明细表和最终任务状态。

### 租约、超时与恢复

第一版默认值：

```text
空队列休眠             2 秒
Adapter 并发           1
任务租约               30 分钟
心跳续租               60 秒
Adapter 最大运行时间   按数据类型配置
最大尝试次数           3
失败退避               1 分钟、5 分钟、30 分钟
```

Worker 启动后先检查租约过期的 `running` 任务：

- 确认没有对应存活子进程后，将任务恢复为 `pending`。
- 清理该任务未完成的 attempt 临时目录。
- 保留原始文件，禁止因 Adapter 失败删除 raw。
- 如果正式产物已经存在但数据库未提交，按稳定业务键校验并 upsert，不能重复生成明细帧。

永久不支持的格式、损坏文件和缺少必要变量属于不可重试错误；进程异常、临时 I/O、数据库短暂不可用属于可重试错误。

## AdapterRunner 约定

Worker 只调用统一入口，不直接了解每个 Adapter 的内部目录和 meta 版本：

```python
run_adapter(
    file_uuid,
    data_type,
    source_path,
    output_root,
    attempt_dir,
    collection_uuid=None,
)
```

当前 Adapter 映射：

```text
CMA       -> adapters/cma_adapter.py
ERA5      -> adapters/era5_adapter.py
GFS       -> adapters/gfs_adapter.py
ECMWF     -> adapters/gfs_adapter.py，data_type=ECMWF
FY3       -> adapters/fy3_adapter.py
Himawari  -> adapters/himawari_adapter.py
Radar     -> adapters/radar_adapter.py
WRF       -> adapters/wrf_adapter.py
```

Himawari 定时下载仍与集合上传 Worker 解耦；用户上传的完整 HSD 集合由 Worker 解析。

统一结果至少包含：

```text
file_uuid
collection_uuid
data_type
meta_path
default_webp_url
webp_count
adapter_name
adapter_version
assets[]
warnings[]
```

FY-3 与 Himawari 的原始文件统一集中到各自的 `data/<类型>/raw/`；日期/时次目录只保存解析后的 WebP 和 `meta/scene.meta.json`。Himawari adapter 已实现 HSD raw 分段下载、断点续传和等经纬度网格重采样。raw 默认保留，不会随服务启动或解析结果自动删除。

所有上传入口使用同名原子覆盖：先写入同目录的 `.upload.part`，完整写入后再替换正式文件。重复上传不会生成 `_1`、`_2` 后缀；上传中断时原正式文件保持不变。
`assets[]` 每行对应一个“要素/产品 × 层次 × 分辨率 × 时次 × WebP”，用于写入对应的数据类型明细表。现有 meta 文件格式不修改；AdapterRunner 负责把不同版本 meta 转换为统一结果。

Adapter 改造要求：

- 原始输入路径和产物输出目录分离，不再默认把产物写在 raw 旁边。
- 同一 `file_uuid + adapter_version` 重试必须得到稳定、可 upsert 的结果。
- 正式产物写入前必须经过临时目录和完整性校验。
- Adapter 内部不领取队列任务，不决定用户权限，不直接修改任务状态。

## 成功与失败提交

子进程成功并通过校验后，主 Worker 在一个数据库事务中：

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `HIMAWARI_AUTO_DOWNLOAD` | `0` | 默认关闭；显式设为 `1` 且提供 FTP 凭据后启动 |
| `HIMAWARI_WINDOW_HOURS` | `1` | 展示窗口和补齐窗口长度 |
| `HIMAWARI_LATEST_DELAY_MINUTES` | `60` | 当前时间向前延迟后作为窗口右边界 |
| `HIMAWARI_DOWNLOAD_INTERVAL_MINUTES` | `10` | 下载时次间隔 |
| `HIMAWARI_DOWNLOAD_INTERVAL_SECONDS` | `60` | 自动任务轮询间隔 |
| `HIMAWARI_DOWNLOAD_MAX_JOBS_PER_RUN` | `7` | 每轮最多处理的时次任务数；`0` 表示不限量 |
| `HIMAWARI_FILE_WORKERS` | `4` | 单个时次 HSD 分段文件并发下载上限 |
| `HIMAWARI_BANDS` | `B13,B03,B02,B01` | 默认下载和解析四个核心通道 |
1. upsert 对应数据类型明细表的 `assets[]`。
2. 更新 `public_info.meta_path`、`default_webp_url`、`webp_count`。
3. 更新 `adapter_name`、`adapter_version`、`parse_finished_at`。
4. 清空 `parse_error`、`parse_worker` 和 `parse_lease_until`。
5. 设置 `parse_status=success`。

失败时：

- 自动窗口右边界为“当前时间 - 60 分钟”的 10 分钟整点，向前覆盖 1 小时。
- 每个时次补齐 `B13/B03/B02/B01`，用于红外显示和真彩色合成。
- 下载、解析失败和解析成功后都保留 `.part/raw`，供续传、复核和重新解析。
- 展示接口按 1 小时时间窗筛选，但不会自动删除过期 raw 或正式解析结果。
- 保存精简、可读的 `parse_error`，详细堆栈写日志。
- 可重试错误计算 `next_parse_at`，设置回 `pending`。
- 不可重试或超过次数设置 `failed`。
- 删除本次未完成产物，保留 raw。
- 前端重试只重置数据库任务，不重复上传原文件。

HSD 原始数据不提交到 Git；需要别人拉取后看到效果时，只提交必要的解析结果 `scene.meta.json + WebP`，不要提交 raw、`.part`、`.float32` 或 `.nc` 中间文件。
文件系统移动和数据库提交无法成为一个原子事务。正式产物路径必须由 `file_uuid` 或稳定 `work_key` 决定，数据库写入使用 upsert，并提供孤儿产物清理任务。

## 启动

当前主后端：

```powershell
cd D:\weather_prediction_system\backend
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8002
```

接口文档：`http://127.0.0.1:8002/docs`

开发环境直接启动 `main.py` 或 Uvicorn 即可，生命周期会自动启动 Worker 子进程：

```powershell
cd D:\weather_prediction_system\backend
.\.venv\Scripts\Activate.ps1
python main.py
```

Worker 不监听端口。FastAPI 关闭或 `--reload` 重载时会同步终止 Worker 及其正在运行的 Adapter 子进程，未完成任务之后通过租约恢复。

生产环境若使用多个 Uvicorn/Gunicorn API 进程，应设置：

```text
START_ADAPTER_WORKER_WITH_API=false
```

然后使用进程管理器单独运行 `python -m workers.parse_worker`。独立命令也保留用于排障；不得同时手动启动多个 Worker，单实例锁会拒绝后启动的进程。

## 第一版验证

当前已使用 GFS 单文件跑通完整链路：

```text
backend_upload 上传 GFS
-> raw 文件存在
-> public_info.pending
-> Worker 领取并设置 running
-> Adapter 子进程生成 meta/WebP
-> gfs_info 写入 208 条要素/时次资源
-> public_info.success
-> /api/display/GFS 仍可展示
```

还需要使用各自真实样本继续回归 ERA5/ECMWF、CMA、Radar、WRF、FY3。自动下载和 Himawari 不在本版本验收范围。

首版至少测试：

- 空队列时 Worker 休眠，任务进入后能够自动领取。
- 当前 Adapter 未完成时不启动第二个任务。
- 意外启动第二个 Worker 时单实例锁生效。
- 子进程成功、普通失败、超时和强制终止。
- Worker 重启后租约过期任务可以恢复。
- 同一任务重试不重复生成明细行。
- raw 文件在失败后仍然存在。
- WebP、meta、数据库明细和 `public_info` 状态一致。
- 前端不再二次上传文件，展示接口响应保持兼容。

## 协作注意

- 不提交大体积气象原始数据、临时分片或 Adapter attempt 目录。
- 小样例放在 `samples/`，用于 Adapter 和 Worker 集成测试。
- `public_info` 是任务状态唯一事实来源，不以目录是否存在推断解析状态。
- 新增 Adapter 必须同时提供结果标准化、明细表映射和失败分类。
- 新增 Python 依赖时同步更新 `requirements.txt`。
- 当前 Himawari 测试调度代码不得作为新 Worker 的复制模板。
