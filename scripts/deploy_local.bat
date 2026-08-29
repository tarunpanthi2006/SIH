@echo off
REM ============================================================
REM SatQuery-RS — Local Deployment Script (Windows)
REM ============================================================
REM Run this on your local machine (RTX 4050 6GB) to start
REM the model server in CPU-offload mode.
REM
REM Usage:
REM   scripts\deploy_local.bat
REM ============================================================

echo ============================================
echo   SatQuery-RS Local Deployment (RTX 4050)
echo ============================================

REM ---- Check .env exists ----
if not exist .env (
    echo.
    echo [ERROR] .env file not found!
    echo Copy .env.example to .env and add your HuggingFace token.
    echo   copy .env.example .env
    exit /b 1
)

REM ---- Download model if needed ----
echo.
echo [1/3] Checking model weights...
python scripts/download_models.py --status

echo.
echo [2/3] Downloading adapter (if available)...
python scripts/download_models.py --adapter 2>nul

REM ---- Start server in CPU-offload mode ----
echo.
echo [3/3] Starting server in CPU-offload mode...
echo.
echo ============================================
echo   Server starting on http://localhost:8100
echo   API docs: http://localhost:8100/docs
echo   Health:   http://localhost:8100/health
echo.
echo   Mode: CPU-offload (safe for 6GB VRAM)
echo   Speed: ~30-60 seconds per query
echo ============================================
echo.

set SATQUERY_MODE=cpu-offload
python -m backend.serve --host 127.0.0.1 --port 8100
