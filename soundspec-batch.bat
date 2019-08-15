@echo off

if "%~1"=="" (goto usage) else (goto execute)
:usage
echo "Usage: %0 C:\path\to\files"
GOTO:EOF

:execute
for %%f in (%1\*) do (
  echo:
  echo %%f
  ffmpeg -loglevel warning -nostdin -i "%%f" "%%f".wav
  soundspec.py -b "%%f".wav
  del "%%f".wav
)
