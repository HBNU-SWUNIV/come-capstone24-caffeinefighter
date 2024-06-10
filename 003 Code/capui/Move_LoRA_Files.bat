@echo off
set "source_folder=Lora"
set "destination_folder=stable-diffusion-webui\models\Lora"

echo Moving files from %source_folder% to %destination_folder%...

if not exist "%destination_folder%" (
    mkdir "%destination_folder%"
)

move "%source_folder%\*" "%destination_folder%"

echo Files moved successfully.
pause