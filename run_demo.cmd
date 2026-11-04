@echo off
rem Starts EVD Studio, the demonstrator, from the evd-vfm folder.
rem Set EVD_PYTHON to the python.exe of the project environment if it is not the default one.
setlocal
cd /d "%~dp0evd-vfm"
if defined EVD_PYTHON goto configured
set "EVD_PYTHON=%USERPROFILE%\miniconda3\envs\pytorch\python.exe"
:configured
if exist "%EVD_PYTHON%" goto launch
echo Python environment not found: %EVD_PYTHON%
echo Set EVD_PYTHON to the python.exe of the project environment.
echo See evd-vfm\apps\demo\README.md for the instructions.
pause
exit /b 1
:launch
set "KMP_DUPLICATE_LIB_OK=TRUE"
"%EVD_PYTHON%" apps\demo\serve.py %*
if not errorlevel 1 exit /b 0
echo.
echo The demonstrator did not start. Read the message above.
pause
