# Weather Data Agent Backend

这是智慧气象系统的独立 Agent 后端，默认端口为 8004，用于接入前端 `src/views/Agent.vue`。

## 功能

- 查询 GFS / ECMWF 展示状态
- 判断当前主展示是否为 WEBP
- 诊断前端不显示问题
- 审计本地资源完整性
- 生成状态报告
- 触发 GFS / ECMWF 下载并解析

## 运行前提

主后端必须先启动：

```bat
cd /d D:\xiazai\python\pythonproject\backend_system
python main.py
```

并确认可访问：

```text
http://127.0.0.1:8002/api/display/ECMWF
```

## 安装依赖

```bat
cd /d D:\xiazai\python\pythonproject\backend_agent
python -m pip install -r requirements.txt
```

## 启动 Agent

```bat
run_agent.bat
```

或手动启动：

```bat
set WEATHER_BACKEND_BASE=http://127.0.0.1:8002
set BACKEND_SYSTEM_DIR=D:\xiazai\python\pythonproject\backend_system
python -m uvicorn main:app --host 127.0.0.1 --port 8004 --reload
```

## 测试

```bat
quick_test.bat
```

或：

```bat
curl -N -X POST http://127.0.0.1:8004/api/agent/chat -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"检查 ECMWF 是否是 WEBP\"}],\"context\":{}}"
```

## 前端可用问题

- 检查 ECMWF 是否是 WEBP
- 查询 GFS 最新数据
- 诊断 ECMWF 为什么不显示
- 审计 ECMWF 资源完整性
- 生成 ECMWF 数据状态报告
- 下载 ECMWF 到 72 小时
- 下载 GFS 到 24 小时，覆盖并解析
