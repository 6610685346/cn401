# Creates "CN401" on the Windows desktop, pointing at "CN401 App.bat".
# Called by setup_and_run.bat (menu 6):
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\make_shortcut.ps1 -Root <V02 folder>
param([Parameter(Mandatory = $true)][string]$Root)
$ErrorActionPreference = 'Stop'

$desktop = [Environment]::GetFolderPath('Desktop')  # may be OneDrive\<localised name>
if (-not $desktop -or -not (Test-Path -LiteralPath $desktop)) {
    $desktop = Join-Path $env:USERPROFILE 'Desktop'
}
if (-not (Test-Path -LiteralPath $desktop)) {
    throw "Desktop folder not found: $desktop"
}

# WScript.Shell cannot save a shortcut to a path containing characters outside
# the system code page (e.g. a Thai desktop folder name), so save it in TEMP
# first, then move it with PowerShell, which handles any file name.
$tmp = Join-Path ([IO.Path]::GetTempPath()) ('cn401_' + [guid]::NewGuid().ToString('N') + '.lnk')
$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($tmp)
$shortcut.TargetPath = Join-Path $Root 'CN401 App.bat'
$shortcut.WorkingDirectory = $Root
$shortcut.WindowStyle = 7  # start the console minimised
$shortcut.Description = 'CN401 person detection'
$shortcut.Save()

$dest = Join-Path $desktop 'CN401.lnk'
Move-Item -LiteralPath $tmp -Destination $dest -Force
Write-Host 'Created "CN401" on the desktop.'
