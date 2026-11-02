<# 首次启动仅在项目内部安装 Python 与依赖，不写入全局 Python 或 Conda 环境。 #>
[CmdletBinding()]
param(
    [string]$TargetDirectory = 'even',
    [switch]$PythonOnly,
    [switch]$ResolveOnly
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $projectRoot
$targetRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot $TargetDirectory))
if (-not $targetRoot.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw '安装目标必须位于当前项目内部。'
}
$uv = '.\tools\bootstrap\uv.exe'
if (-not (Test-Path -LiteralPath $uv -PathType Leaf)) { throw '缺少随程序包提供的 tools/bootstrap/uv.exe。' }
New-Item -ItemType Directory -Path '.\cache\uv', '.\cache\python', '.\cache\tmp', '.\logs' -Force | Out-Null
$env:UV_CACHE_DIR = (Resolve-Path '.\cache\uv').Path
$env:TEMP = (Resolve-Path '.\cache\tmp').Path
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'

try {
    $python = Join-Path $targetRoot 'python.exe'
    $pending = Join-Path $targetRoot '.indextts_setup_pending'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        Write-Host '[1/3] 下载项目内 Python 3.11，完成后复制到 even/。'
        & $uv python install '3.11' --install-dir '.\cache\python' --no-bin --no-registry --no-config
        if ($LASTEXITCODE -ne 0) { throw 'Python 下载失败，请检查网络后再次运行 run.bat。' }
        $candidates = @(Get-ChildItem -LiteralPath '.\cache\python' -Directory |
            Where-Object { $_.Name -like 'cpython-3.11.*-windows-x86_64-*' -and (Test-Path -LiteralPath (Join-Path $_.FullName 'python.exe')) } |
            Sort-Object Name -Descending)
        if ($candidates.Count -eq 0) { throw 'Python 下载完成，但没有找到可用解释器。' }
        New-Item -ItemType Directory -Path $targetRoot -Force | Out-Null
        Get-ChildItem -LiteralPath $candidates[0].FullName -Force | Copy-Item -Destination $targetRoot -Recurse -Force
        [IO.File]::WriteAllText($pending, 'Dependency installation is pending.', [Text.UTF8Encoding]::new($false))
    }
    # Windows PowerShell 5.1 调用原生命令时会移除 -c 内容中的引号，验证表达式避免内嵌字符串。
    & $python -X utf8 -I -B -c 'import sys; assert sys.version_info[:2] == (3,11); print(sys.version.split()[0], sys.executable)'
    if ($LASTEXITCODE -ne 0) { throw '复制后的项目内 Python 验证失败。' }
    # 缓存中的 Python 由 uv 管理；复制到项目 even/ 后由本项目独立维护其依赖。
    $managementMarker = Join-Path $targetRoot 'Lib\EXTERNALLY-MANAGED'
    if (Test-Path -LiteralPath $managementMarker) { Remove-Item -LiteralPath $managementMarker -Force }
    if ($PythonOnly) { Write-Host '项目内 Python 安装与独立运行检查通过。'; exit 0 }
    Write-Host '[2/3] 安装语音合成依赖到 even/。CUDA 依赖较大，首次安装需要联网。'
    $installed = $false
    # 普通依赖优先清华镜像，失败后切换阿里云；CUDA wheel 使用 requirements 中的 PyTorch 源。
    foreach ($mirror in @('https://pypi.tuna.tsinghua.edu.cn/simple', 'https://mirrors.aliyun.com/pypi/simple/')) {
        $installArgs = @('pip', 'install', '--python', $python, '--requirement', '.\requirements.txt',
            '--link-mode', 'copy', '--index-strategy', 'unsafe-best-match', '--index-url', $mirror, '--no-config')
        if ($ResolveOnly) { $installArgs += '--dry-run' }
        & $uv @installArgs
        if ($LASTEXITCODE -eq 0) { $installed = $true; break }
        Write-Host '当前镜像安装未完成，尝试备用镜像；已下载的内容会保留在项目缓存中。'
    }
    if (-not $installed) { throw '依赖安装失败，日志和下载缓存已保留；再次运行 run.bat 可重试。' }
    if ($ResolveOnly) { Write-Host '完整依赖解析检查通过；未执行依赖安装。'; exit 0 }
    Write-Host '[3/3] 检查依赖完整性。'
    & $python -X utf8 -I -B -m pip check
    if ($LASTEXITCODE -ne 0) { throw '依赖完整性检查未通过。' }
    if (Test-Path -LiteralPath $pending) { Remove-Item -LiteralPath $pending -Force }
    Write-Host '项目内环境安装完成。模型需要另行放入 models/IndexTTS-2/。'
    exit 0
} catch {
    Write-Error $_
    exit 1
}
