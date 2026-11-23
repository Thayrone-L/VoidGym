@echo off
setlocal
cd /d "%~dp0"
set "TRAY=tools\VoidGymTray\bin\Release\net10.0-windows\VoidGymTray.exe"
if exist "%TRAY%" (
  start "" "%TRAY%"
) else (
  dotnet run --project tools\VoidGymTray\VoidGymTray.csproj -c Release
)
