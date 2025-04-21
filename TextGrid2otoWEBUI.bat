@echo off
chcp 65001 >nul

set PYTHON_REL_PATH=workenv\python.exe
set SCRIPT_REL_PATH=WEB-UI.py
set REQUIREMENTS_FILE=requirements.txt

set SCRIPT_DIR=%~dp0
set PYTHON_PATH=%SCRIPT_DIR%%PYTHON_REL_PATH%
set SCRIPT_PATH=%SCRIPT_DIR%%SCRIPT_REL_PATH%

echo 启动 TextGrid2oto-WEBUI
"%PYTHON_PATH%" "%SCRIPT_PATH%"
pause
