@echo off
rem CN401 desktop app - double-click to open.
rem First run sets everything up (can take 10+ minutes), later runs are quick.
cd /d "%~dp0"
call "%~dp0setup_and_run.bat" app
