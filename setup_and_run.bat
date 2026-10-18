@echo off
setlocal
rem ==========================================================================
rem  CN401 person detection - one-click setup + run (Windows Command Prompt)
rem
rem  Double-click this file, or from Command Prompt:
rem      setup_and_run.bat          setup, then show the menu
rem      setup_and_run.bat app      setup, then open the desktop app (no menu)
rem      setup_and_run.bat run      setup, then start detection straight away
rem      setup_and_run.bat web      setup, then start the LAN history page only
rem
rem  "CN401 App.bat" (or the desktop shortcut from menu 6) runs "app".
rem
rem  First run : creates .venv, installs requirements.txt, downloads models.
rem  Later runs: skips whatever is already done (fast).
rem  Re-installs packages automatically when requirements.txt changes.
rem ==========================================================================

cd /d "%~dp0"
set "ROOT=%CD%"
set "APP=%ROOT%\new"
set "MODELS=%APP%\models"
set "VENV=%ROOT%\.venv"
set "STAMP=%VENV%\requirements.installed"

if not exist "%APP%\connect.py" (
    echo Cannot find new\connect.py next to this file.
    echo Put setup_and_run.bat in the V01 folder.
    goto :fail
)

rem ---------------------------------------------------------------- 1. Python
echo [1/4] Checking Python ...
set "PY="
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if defined PY goto :check_version
where python >nul 2>nul
if not errorlevel 1 set "PY=python"
if not defined PY goto :no_python

:check_version
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 goto :no_python

if exist "%VENV%\Scripts\python.exe" goto :venv_ready
echo       Creating virtual environment in .venv ...
%PY% -m venv "%VENV%"
if errorlevel 1 goto :fail

:venv_ready
call "%VENV%\Scripts\activate.bat"
if errorlevel 1 goto :fail
python --version

rem ---------------------------------------------------------- 2. Requirements
if not exist "%STAMP%" goto :install
fc /b "%ROOT%\requirements.txt" "%STAMP%" >nul 2>nul
if errorlevel 1 goto :install
echo [2/4] Packages already installed.
goto :models

:install
echo [2/4] Installing packages (first time can take 10+ minutes) ...
python -m pip install --upgrade pip
python -m pip install -r "%ROOT%\requirements.txt"
if errorlevel 1 goto :fail
copy /y "%ROOT%\requirements.txt" "%STAMP%" >nul

rem ---------------------------------------------------------------- 3. Models
:models
echo [3/4] Checking models in new\models ...
if not exist "%MODELS%" mkdir "%MODELS%"

set "F=%MODELS%\yolo11n-pose.pt"
if exist "%F%" goto :yolo_ok
echo       Downloading yolo11n-pose.pt ...
curl -L --fail -o "%F%" https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt
if errorlevel 1 goto :fail_download
:yolo_ok

set "F=%MODELS%\face_detection_yunet_2023mar.onnx"
if exist "%F%" goto :yunet_ok
echo       Downloading face_detection_yunet_2023mar.onnx ...
curl -L --fail -o "%F%" https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
if errorlevel 1 goto :fail_download
:yunet_ok

set "F=%MODELS%\face_recognition_sface_2021dec.onnx"
if exist "%F%" goto :sface_ok
echo       Downloading face_recognition_sface_2021dec.onnx ...
curl -L --fail -o "%F%" https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
if errorlevel 1 goto :fail_download
:sface_ok

rem config.py reads models\vlm_par.pt (how_to_use.txt says best_model.pt).
set "F=%MODELS%\vlm_par.pt"
if exist "%F%" goto :vlm_ok
if exist "%MODELS%\best_model.pt" (
    echo       Using existing best_model.pt as vlm_par.pt
    copy /y "%MODELS%\best_model.pt" "%F%" >nul
    goto :vlm_ok
)
echo       Downloading VLM-PAR weights from Hugging Face (large) ...
python -c "import shutil, sys; from huggingface_hub import hf_hub_download; shutil.copy(hf_hub_download('Bekhouche/VLM-PAR', filename='best_model.pt'), sys.argv[1])" "%F%"
if errorlevel 1 goto :fail_download
:vlm_ok

rem An NVIDIA card with the CPU build of PyTorch is the most common slow-down.
where nvidia-smi >nul 2>nul
if errorlevel 1 goto :gpu_done
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
if not errorlevel 1 goto :gpu_done
echo.
echo   NOTE: an NVIDIA GPU is installed but PyTorch cannot use it (CPU build).
echo         Everything works, just slower. See "GPU" in how _to_use.txt.
:gpu_done

echo [4/4] Ready.
if /i "%~1"=="app" goto :app_once
if /i "%~1"=="run" goto :run_once
if /i "%~1"=="web" goto :web

rem ------------------------------------------------------------------- menu
:menu
echo.
echo   1  Desktop app                (cameras, people, settings - main program)
echo   2  Run detection              (connect.py, one camera in an OpenCV window)
echo   3  Add a person               (capture face photos, then enroll)
echo   4  Enroll from saved photos   (new\models\photos\NAME)
echo   5  LAN history page only      (when the desktop app is closed)
echo   6  Create a desktop shortcut  (CN401 on the Windows desktop)
echo   Q  Quit
set "CHOICE="
set /p "CHOICE=Choose 1-6/Q: "
if /i "%CHOICE%"=="1" goto :app
if /i "%CHOICE%"=="2" goto :run
if /i "%CHOICE%"=="3" goto :add_person
if /i "%CHOICE%"=="4" goto :enroll_only
if /i "%CHOICE%"=="5" goto :web
if /i "%CHOICE%"=="6" goto :shortcut
if /i "%CHOICE%"=="q" goto :end
goto :menu

rem ------------------------------------------------------------ desktop app
rem pythonw = no console window; messages go to logs\app.log
:app
start "" "%VENV%\Scripts\pythonw.exe" "%ROOT%\desktop.py"
echo The app is opening in its own window (log: logs\app.log).
goto :menu

:app_once
start "" "%VENV%\Scripts\pythonw.exe" "%ROOT%\desktop.py"
goto :end

:shortcut
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\tools\make_shortcut.ps1" -Root "%ROOT%"
if errorlevel 1 echo Could not create the shortcut, see the error above.
goto :menu

:run
pushd "%APP%"
python connect.py
popd
goto :menu

:run_once
pushd "%APP%"
python connect.py
popd
goto :end

:add_person
set "NAME="
set /p "NAME=Person name: "
if not defined NAME goto :menu
pushd "%APP%"
echo Press SPACE to save a photo when the green face box shows, Q when done.
python capture_faces.py "%NAME%"
python enroll.py "%NAME%" "models\photos\%NAME%"
popd
goto :menu

:enroll_only
set "NAME="
set /p "NAME=Person name (folder under new\models\photos): "
if not defined NAME goto :menu
pushd "%APP%"
python enroll.py "%NAME%" "models\photos\%NAME%"
popd
goto :menu

rem ------------------------------------------------------ LAN history page
rem The desktop app serves this page itself while it is open; use this only
rem when the app is closed. Accounts are made in the app: Admin page -> Users.
:web
if not exist "%ROOT%\web\manage.py" (
    echo Cannot find web\manage.py
    goto :menu
)
pushd "%ROOT%\web"
python manage.py migrate --noinput
if errorlevel 1 goto :web_failed
rem First time only: an account to log in to the history page.
python manage.py shell -c "import sys; from django.contrib.auth.models import User; sys.exit(0 if User.objects.exclude(username='cn401-app').exists() else 3)" >nul 2>nul
if errorlevel 3 (
    echo.
    echo Create an account for the history page:
    python manage.py createsuperuser
)
python manage.py runportal
popd
goto :menu

:web_failed
popd
echo Web portal setup failed, see the error above.
goto :menu

rem ----------------------------------------------------------------- errors
:fail_download
echo.
echo *** Download failed: %F%
echo     Check the internet connection, then run this file again.
if exist "%F%" del "%F%"
goto :fail

:no_python
echo.
echo *** Python 3.10 or newer was not found.
echo     Install it from https://www.python.org/downloads/
echo     and tick "Add python.exe to PATH" during install.
goto :fail

:fail
echo.
echo Setup stopped. Fix the error above and run this file again.
pause
endlocal
exit /b 1

:end
endlocal
exit /b 0
