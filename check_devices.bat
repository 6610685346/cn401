@echo off
rem CN401 - check this computer (GPU, cameras, IP cameras, network, devices on the LAN).
rem Portable: copy check_devices.bat + check_devices.py to any Windows PC and double-click.
rem Needs only Python 3.8+ (no .venv, no project folder). Nothing is installed or changed.
rem   check_devices.bat --quick                          do not open the cameras
rem   check_devices.bat --no-lan                         do not look for devices on the LAN
rem   check_devices.bat --rtsp "rtsp://user:pass@IP:554/stream1"   also test this camera
setlocal
cd /d "%~dp0"
chcp 65001 >nul
if not exist "%~dp0check_devices.py" (
    echo check_devices.py must be in the same folder as this file.
    goto :end
)
rem Any Python works. The CN401 .venv (if this is the project folder) is used when it is
rem complete, because it also has PyTorch / OpenCV and shows a little more.
if exist "%~dp0.venv\Scripts\python.exe" if exist "%~dp0.venv\Scripts\activate.bat" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0check_devices.py" %*
    goto :end
)
where py >nul 2>nul
if not errorlevel 1 (
    py -3 "%~dp0check_devices.py" %*
    if not errorlevel 9009 goto :end
)
where python >nul 2>nul
if not errorlevel 1 (
    python "%~dp0check_devices.py" %*
    goto :end
)
echo Python not found on this computer.
echo Install Python 3.8 or newer from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH"), then run this file again.
:end
echo.
pause
