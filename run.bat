@echo off
setlocal
chcp 65001 >nul
rem 进入当前项目，全部程序与运行环境均从相对路径加载。
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONPATH=
set PYTHONHOME=
set "PATH=%CD%\even;%CD%\even\Library\bin;%CD%\even\Scripts;%PATH%"
if not exist "even\python.exe" (
    call :setup_environment
    if errorlevel 1 exit /b 1
)
if exist "even\.indextts_setup_pending" (
    call :setup_environment
    if errorlevel 1 exit /b 1
)
rem 即使 even 已存在，也要确保安装了台语语言包新增依赖。
findstr /x /c:"1.4.0" "even\.indextts_environment_version" >nul 2>nul
if errorlevel 1 (
    call :setup_environment
    if errorlevel 1 exit /b 1
)
"even\python.exe" -X utf8 -I -B "start.py" %*
set "INDEXTTS_EXIT=%ERRORLEVEL%"
if not "%INDEXTTS_EXIT%"=="0" (
    echo [错误] 启动失败，详细信息请查看 logs\indextts.log。
    pause
)
exit /b %INDEXTTS_EXIT%

:setup_environment
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "tools\setup_environment.ps1"
if errorlevel 1 (
    echo [错误] 项目内环境安装未完成，请检查网络及上述错误后重试。
    pause
    exit /b 1
)
exit /b 0
