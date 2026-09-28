@echo off
chcp 65001 >nul
cd /d "%~dp0"
start "AI面试助手服务" cmd /k python app.py
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8765"
