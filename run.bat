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
    echo [错误] 缺少项目内 even\python.exe，请恢复项目运行环境。
    pause
    exit /b 1
)
"even\python.exe" -X utf8 -I -B "start.py" %*
set "INDEXTTS_EXIT=%ERRORLEVEL%"
if not "%INDEXTTS_EXIT%"=="0" (
    echo [错误] 启动失败，详细信息请查看 logs\indextts.log。
    pause
)
exit /b %INDEXTTS_EXIT%
