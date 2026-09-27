@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "RAG_PROJECT=%~dp0projects\research_assistant"
set "RAG_PYTHON=%RAG_PROJECT%\.venv\Scripts\python.exe"

if not exist "%RAG_PYTHON%" (
  echo Python environment not found: %RAG_PYTHON%
  echo Create the virtual environment under projects\research_assistant\.venv first.
  pause
  exit /b 2
)

cd /d "%RAG_PROJECT%"
"%RAG_PYTHON%" -X utf8 personal_live_web.py %*
set "RAG_EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%RAG_EXIT_CODE%"=="0" echo Startup or runtime failed with code %RAG_EXIT_CODE%.
pause
exit /b %RAG_EXIT_CODE%
