@echo off
rem Starts Forma, the spatial annotation tool, from the evd-vfm folder.
rem The first start creates the local project in runs\annotazioni_spaziali.
setlocal
cd /d "%~dp0evd-vfm"
if defined EVD_PYTHON goto configured
set "EVD_PYTHON=%USERPROFILE%\miniconda3\envs\pytorch\python.exe"
:configured
if not exist "%EVD_PYTHON%" set "EVD_PYTHON=python"
if exist "runs\annotazioni_spaziali\project.json" (
    "%EVD_PYTHON%" scripts\annota_spaziale.py
) else (
    "%EVD_PYTHON%" scripts\annota_spaziale.py --init-local
)
if errorlevel 1 pause
endlocal
