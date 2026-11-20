# 季节性降水产品契约

第一版通过 `backend_upload` 上传 NetCDF/GRIB/GRIB2，数据库数据类型使用
`SEASONAL_PRECIP`（发布目录名 `SeasonalPrecip`）。Adapter 将产品转成
WebP 与 `schema_version=2.0` 的 `meta.json`，前端复用现有 `GribLayer`。

## 支持的变量

变量名或属性中包含以下语义即可被识别：

- 降水量：`precip`、`rain`、`tp`、`apcp`、`pr`、`ppt`、`prate`；单位会尽量规范为 `mm`。
- 距平：`anomaly`、`anom`、`departure`、`percentile`；可使用 `mm` 或 `%`。
- 概率：`probability`、`below`、`normal`、`above`、`tercile`；0–1 的值会转换成 0–100%。

推荐使用二维经纬度网格，维度形如 `time/lead × lat × lon`。经纬度坐标名支持
`lat/latitude/y` 和 `lon/longitude/x`。额外的时间或 lead 维度会转换为地图帧。
如果概率变量带有 `category`、`tercile` 或 `class` 维度，Adapter 会将每个类别拆成独立图层。

## 元数据

产品名优先读取全局属性 `title`、`product`、`model`，提供方读取
`provider`、`institution` 或 `source`。后续模型服务只要输出相同的变量层和
`meta.json` 字段，就可以直接复用上传、目录和前端地图链路。

经纬度范围只用于前端地图视野，第一版不裁剪原始文件。

## 公共自动下载产品

上传服务内置 NOAA/NCEI NMME CFSv2 产品源。`GET http://127.0.0.1:8003/api/seasonal/products` 返回当前可下载的 `pr_6hour_cfsv2-2011...nc` 文件，`POST /api/seasonal/download` 下载并写入共享解析队列。NMME 的 `pr` 变量是降水率，单位 `kg m-2 s-1`，适配器转换为 `mm/day` 后展示。生产环境应保留 NOAA/NCEI/NMME 数据致谢。

## 接口与权限

- `POST /api/upload/chunk`、`POST /api/upload/complete`：`role >= 2`，上传并入队。
- `GET /api/upload/tasks`：`role >= 1`，查看解析状态。
- `GET /api/catalog/resources?data_type=SEASONAL_PRECIP`：`role >= 1`，列出可选产品。
- `GET /api/catalog/resources/{file_uuid}`：`role >= 1`，读取产品完整元数据。
- `GET /api/display/SEASONAL_PRECIP`：`role >= 1`，读取最新产品兼容展示响应。
- `GET /api/seasonal/products`：`role >= 1`，读取 NOAA/NCEI 在线产品目录。
- `POST /api/seasonal/download`：`role >= 1`，下载指定 NMME CFSv2 文件并进入后台解析队列。

前端产品选择使用目录资源的 `file_uuid`，不会把用户再次上传的文件直接提交到
8002。上传完成后轮询 `parse_status`，只有成功的资源才进入产品列表。
