@echo off
chcp 65001 >nul
setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo [1/4] Зависимости...
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller || goto :err

echo [2/4] Модели OCR...
python download_models.py || goto :err

echo [3/4] Сборка exe...
pyinstaller --noconfirm --clean --onefile --windowed ^
  --name DocumentOCR ^
  --add-data "ocr_models;ocr_models" ^
  --collect-all rapidocr_onnxruntime ^
  --collect-submodules onnxruntime ^
  --collect-data onnxruntime ^
  --hidden-import onnxruntime.capi._pybind_state ^
  --hidden-import pymupdf ^
  --hidden-import openpyxl ^
  --hidden-import PIL._imaging ^
  --exclude-module torch --exclude-module torchvision ^
  --exclude-module easyocr --exclude-module paddle --exclude-module paddleocr ^
  --exclude-module tkinter --exclude-module matplotlib ^
  main.py || goto :err

echo [4/4] Копирование моделей рядом с exe...
if not exist dist\ocr_models mkdir dist\ocr_models
xcopy /E /I /Y ocr_models dist\ocr_models >nul

echo.
echo ГОТОВО: dist\DocumentOCR.exe
pause
exit /b 0
:err
echo ОШИБКА сборки.
pause
exit /b 1
