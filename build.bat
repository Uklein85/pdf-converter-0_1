@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo ============================================================
echo  DocumentOCR - сборка exe (АО "СТНГ", Уклеин И.В., 2026)
echo ============================================================

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python не найден в PATH. Установите Python 3.10+ x64.
    pause & exit /b 1
)

echo [1/6] Зависимости...
python -m pip install --upgrade pip setuptools wheel || goto :err
python -m pip install -r requirements.txt || goto :err
python -m pip install "pyinstaller>=6.6" || goto :err

echo [2/6] Модели OCR...
python prepare_models.py || goto :err

echo [3/6] Проверка синтаксиса...
python -m py_compile main.py pipeline.py gpr_core.py pdf_io.py exporters.py ocr_engines.py || goto :err

echo [4/6] Smoke-тест (OCR + таблицы ГПР + экспорт)...
python smoke_test.py || goto :err

echo [5/6] Сборка exe (2-6 минут)...
if exist build rmdir /s /q build
pyinstaller --noconfirm --clean DocumentOCR.spec || goto :err

echo [6/6] Модели рядом с exe...
if exist dist\ocr_models rmdir /s /q dist\ocr_models
xcopy /E /I /Y ocr_models dist\ocr_models >nul

echo.
echo ============================================================
if exist dist\DocumentOCR.exe (
    for %%A in ("dist\DocumentOCR.exe") do echo  ГОТОВО: dist\DocumentOCR.exe  (%%~zA байт^)
) else (
    echo  ГОТОВО: dist\DocumentOCR\DocumentOCR.exe  (onedir^)
)
echo  Запустите с консоли, чтобы видеть ошибки:
echo    dist\DocumentOCR.exe
echo ============================================================
pause
exit /b 0

:err
echo.
echo [ERROR] Сборка остановлена. Смотрите сообщение выше.
pause
exit /b 1
