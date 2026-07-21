# ERA5 10 米风场数据契约（R5 第一阶段）

ERA5 adapter 仅在同一数据集内找到兼容的 `u10` 和 `v10` 时生成风场资产。风场生成失败不会阻断原有 WebP 和元数据解析；调用方应先检查 `wind_field.available`。

## 二进制文件

每个时次分别生成两个无文件头文件：

```text
<source_stem>_u10_step000.float32
<source_stem>_v10_step000.float32
```

- 类型：IEEE 754 Float32
- 字节序：little-endian
- 数组序：C / row-major
- 行方向：北到南
- 列方向：西到东
- 格点位置：cell center
- 缺测值：`-999999.0`
- 文件大小：`width * height * 4` 字节
- U 为正表示向东，V 为正表示向北
- 任一分量缺测时，该格点的 U、V 都写为缺测值

R5 只输出原生 ERA5 风场网格，不为 1 km/3 km WebP 插值层生成 U/V 文件。

## 可用风场元数据

```json
{
  "wind_field": {
    "schema_version": "1.0",
    "available": true,
    "product": "10m_wind",
    "components": {"u": "u10", "v": "v10"},
    "level": "10 m above ground",
    "unit": "m/s",
    "source_units": {"u": "m s**-1", "v": "m s**-1"},
    "speed_variable": "ws10",
    "display_range": {"min": 0.0, "max": 30.0},
    "palette": ["#2563eb", "#0891b2", "#16a34a", "#facc15", "#dc2626"],
    "times": ["2025-07-01T08:00"],
    "grid": {
      "crs": "EPSG:4326",
      "width": 1440,
      "height": 721,
      "extent": [-180.0, -90.0, 179.75, 90.0],
      "origin": "north_west",
      "scan_order": "row_major",
      "row_order": "north_to_south",
      "column_order": "west_to_east",
      "grid_registration": "cell_center",
      "lon_step": 0.25,
      "lat_step": 0.25,
      "periodic_longitude": true
    },
    "encoding": {
      "dtype": "float32",
      "byte_order": "little",
      "layout": "component_separated",
      "array_order": "C",
      "bytes_per_value": 4,
      "nodata": -999999.0,
      "invalid_when_either_component_is_nodata": true
    },
    "frames": [
      {
        "index": 0,
        "time": "2025-07-01T08:00",
        "u_url": "/data/ERA5/example_u10_step000.float32",
        "v_url": "/data/ERA5/example_v10_step000.float32",
        "speed_webp_url": "/data/ERA5/example_ws10_step000.webp",
        "component_byte_length": 4152960,
        "u_min": -10.0,
        "u_max": 12.0,
        "v_min": -8.0,
        "v_max": 9.0,
        "speed_min": 0.0,
        "speed_max": 15.0,
        "speed_mean": 6.5
      }
    ]
  }
}
```

同一批 URL 也会写入 `variables[].float32.paths`、`variable_layers.<component>.float32_urls` 和对应的 `native.float32_urls`。

## 合成风速与显示一致性

当 U/V 风场可用时，adapter 额外生成派生变量 `ws10 = sqrt(u10² + v10²)`：

- `ws10` 只表示风速大小，因此统计值和图例从 `0 m/s` 开始，不会出现负数。
- 每个时次生成 `<source_stem>_ws10_stepNNN.webp`，所有时次共用 `wind_field.display_range` 和 `wind_field.palette`。
- 粒子方向和位移来自同一时次的 U/V Float32；粒子颜色和 `ws10` WebP 使用同一风速、显示范围和五段色带。
- 前端仅在选中 `wind_field.speed_variable`（当前为 `ws10`）时加载并显示粒子。U/V 仍作为带正负号的方向分量单独展示。

第一阶段不扩展 SQLite 表；现有 `era5_layer_asset` 仍只登记 WebP。风场文件以 meta 中的 `wind_field.frames` 为唯一配对清单。

## 不可用风场

不可用时保留稳定结构：

```json
{
  "wind_field": {
    "schema_version": "1.0",
    "available": false,
    "product": "10m_wind",
    "components": {"u": "u10", "v": null},
    "reason": "missing_components",
    "detail": {"missing": ["v"]}
  }
}
```

可能的 `reason` 包括：

- `missing_components`
- `ambiguous_components`
- `time_dimension_mismatch`
- `time_count_mismatch`
- `unsupported_time_dimensions`
- `unsupported_component_dimensions`
- `incompatible_units`
- `grid_shape_mismatch`
- `grid_coordinate_mismatch`
- `wind_grid_too_small`
- `wind_grid_scan_order_invalid`
- `wind_grid_not_regular`
- `grid_changed_between_frames`
- `float32_byte_length_mismatch`
- `wind_field_generation_failed`

## 展示接口

`GET /api/display/ERA5` 在响应的 `data.wind_field` 顶层直接返回经过检查的风场描述，前端不需要读取整份 `meta_json`。`data.meta_json.wind_field` 会使用同一份规范化结果。

展示服务会进行以下检查：

- 网格宽高、范围、编码和每分量字节数符合本契约
- `frames` 与 `times` 数量、索引和时间逐帧一致
- U/V 文件只允许位于本机 ERA5 静态目录
- 文件存在且大小严格等于 `width * height * 4`
- 返回 URL 统一为 `/data/ERA5/<relative-path>.float32`

旧 meta 没有 `wind_field` 时返回：

```json
{
  "available": false,
  "reason": "not_provided"
}
```

展示契约或文件检查失败时，整组风场降级为：

```json
{
  "available": false,
  "reason": "display_contract_invalid",
  "detail": {
    "code": "asset_missing",
    "frame_index": 0
  }
}
```

该降级只关闭风场能力，不影响同一响应中的 WebP、变量、分辨率和时间信息。
