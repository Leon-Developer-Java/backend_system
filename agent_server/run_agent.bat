@echo off
cd /d %~dp0
set WEATHER_BACKEND_BASE=http://127.0.0.1:8002
set BACKEND_SYSTEM_DIR=%~dp0..
python -m uvicorn main:app --host 127.0.0.1 --port 8004 --reload
