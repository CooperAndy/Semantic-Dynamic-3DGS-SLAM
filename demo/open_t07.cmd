@echo off
setlocal
cd /d "%~dp0"
set "T07_PY="
python --version >nul 2>&1
if not errorlevel 1 set "T07_PY=python"
if not defined T07_PY (
  py -3 --version >nul 2>&1
  if not errorlevel 1 set "T07_PY=py -3"
)
if not defined T07_PY if exist "C:\anaconda\python.exe" set "T07_PY="C:\anaconda\python.exe""
if not defined T07_PY (
  echo [ERROR] Python 3 was not found.
  pause
  exit /b 1
)
%T07_PY% "%~dp0launch_t07.py" --background %*
set "T07_EXIT=%ERRORLEVEL%"
if not "%T07_EXIT%"=="0" pause
exit /b %T07_EXIT%
