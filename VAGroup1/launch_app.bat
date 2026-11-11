@echo off
title Grocery Promotion & Stockpiling Explorer

:: ── Locate Python ──────────────────────────────────────────────────────────
set PYTHON=C:\Users\Tjedw\AppData\Local\Programs\Python\Python312\python.exe

if not exist "%PYTHON%" (
    echo [ERROR] Python 3.12 not found at:
    echo   %PYTHON%
    echo Please update the PYTHON variable in this batch file.
    pause
    exit /b 1
)

:: ── Set working directory to the repo folder ───────────────────────────────
cd /d "%~dp0"

:: ── Confirm Streamlit is installed ─────────────────────────────────────────
"%PYTHON%" -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Streamlit is not installed. Installing now...
    "%PYTHON%" -m pip install streamlit
)

:: ── Optional: set a Gemini API key so the AI narrator can call the LLM ─────
:: Remove the leading "::" to activate, then paste your actual key after the =
:: set GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE

:: ── Launch the app ──────────────────────────────────────────────────────────
echo.
echo  ============================================================
echo   Grocery Promotion ^& Stockpiling Explorer
echo   AI Insight Narrator: ACTIVE
echo   Opening at http://localhost:8501
echo  ============================================================
echo.

"%PYTHON%" -m streamlit run app.py --server.port 8501 --server.headless false

pause

