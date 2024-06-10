@echo off
set "current_directory=%cd%"
set "destination_folder=venv\Lib\site-packages\webuiapi"

echo Moving webuiapi.py from %current_directory% to %destination_folder%...

move "%current_directory%\webuiapi.py" "%destination_folder%"

echo webuiapi.py moved successfully.
pause
