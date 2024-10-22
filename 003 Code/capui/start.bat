@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--xformers --medvram --api --port=7861 --nowebui

cd stable-diffusion-webui

start webui.bat

cd ../

call venv\Scripts\activate.bat

python gradio_ui.py

pause