@echo off
set "current_directory=%cd%"
set "destination_folder=stable-diffusion-webui"

echo Moving config.json from %current_directory% to %destination_folder%...

move "%current_directory%\config.json" "%destination_folder%"

echo config.json moved successfully.
pause
