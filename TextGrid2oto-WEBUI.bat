@echo off
:: Check for admin rights


set PYTHON_REL_PATH=.\workenv\python.exe
set SCRIPT_REL_PATH=WEB-UI.py

set SCRIPT_DIR=%~dp0
set PYTHON_PATH=%SCRIPT_DIR%%PYTHON_REL_PATH%
set SCRIPT_PATH=%SCRIPT_DIR%%SCRIPT_REL_PATH%

cd /d %SCRIPT_DIR%

echo Æô¶¯TextGrid2oto-WEBUI
"%PYTHON_PATH%" "%SCRIPT_PATH%"
pause
