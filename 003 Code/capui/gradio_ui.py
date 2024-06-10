import gradio as gr
import txt2img
import img2img
# import inpaint
import os
import img_viewer
import background_remover
import api_client

##
## 아래 코드들 사용해보기
## 
## 아래 코드를 이용해 progress bar 만들어보기
## def progress():
##     return api.util_wait_for_ready()
##
## txt2img.py에서 이미지를 생성 버튼을 누를 때마다 모델이 바뀌는 것이 아닌 드롭다운에서 모델을 선택하면 바뀌는 것으로 아래 코드를 이용하여 수정
## api_client.api.util_set_model()
##
## 아래 코드를 이용하여 model refresh 구현해보기
## api_client.api.refresh_checkpoints()
##
## Unity Folder로 이미지 이동하는 코드 추가 필요
##

def add_loras(lora_name):
    lora = "<lora:" + lora_name + ":1>"
    return lora

def open_folder(model_input_path):
    model_input_path = "stable-diffusion-webui\models\Stable-diffusion"
    try:
        os.system(f'explorer "{model_input_path}"')
        return
    except Exception as e:
        return f"오류가 발생했습니다: {e}"
    
def open_image_folder(model_input_path):
    model_input_path = "stable-diffusion-webui\output"
    try:
        os.system(f'explorer "{model_input_path}"')
        return
    except Exception as e:
        return f"오류가 발생했습니다: {e}"

def open_removed_folder(model_input_path):
    model_input_path = "background-removed-images"
    try:
        os.system(f'explorer "{model_input_path}"')
        return
    except Exception as e:
        return f"오류가 발생했습니다: {e}"

def get_model_names(modelfolder_path, extensions):
    try:
        # 폴더 내의 모든 파일과 폴더를 리스트로 가져옵니다.
        all_files = os.listdir(modelfolder_path)
        # 지정된 확장자를 가진 파일들을 필터링합니다.
        model_names = [file for file in all_files if file.endswith(extensions)]
        return model_names
    except Exception as e:
        return f"오류가 발생했습니다: {e}"

lorafolder_path = "stable-diffusion-webui\models\Lora"
extensions = (".ckpt", ".safetensors")

model_names = api_client.api.util_get_model_names()
current_model = api_client.api.util_get_current_model()
# lora_names = get_model_names(lorafolder_path, extensions)
lora_names = api_client.api.util_get_lora_names()

lora_list = [
        ('None', ''), 
        ('Texture_1', 'texture,  <lora:texture_1:1>'), 
        ('Sprite_1_epoch_1', 'sprite,  <lora:sprite_1-000001:0.8>'), 
        ('Sprite_1_epoch_2', 'sprite,  <lora:sprite_1:0.8>'), 
        ('Texture', 'diffuse texture, <lora:DiffuseTexture_v11:1>'), 
        ('Metal Texture', 'metal texture, <lora:dirtymetal_textures_1:0.8>'), 
        ('Old school Texture', 'texture, old school, quake, <lora:Quake_Lora:1>'), 
        ('Book', 'book, <lora:FantasyIcons_Books_noFlip:1>'), 
        ('Gemstone', 'gemstone, <lora:FantasyIcons_Gemstones:1>'), 
        ('pixel sprites', 'pixel, pixel art, pixelart, xiangsu, xiang su, <lora:pixel sprites:1>'), 
        ('PixelAnimal', 'animal, pixel, pixel art, pixelart, xiangsu, xiang su, <lora:PixelAnimal:1>'), 
        ('Pixel Weapon(axe, sword, bow)', 'weapon, no humans, pixel, pixel art, pixelart, <lora:pixel sword:1>'), 
        ('Pixel Gun', 'gun, no humans, pixel, pixel art, pixelart, <lora:pixel gun:1>'), 
        ('Pixel Book', 'boox,pixel, pixel art, pixelart, xiangsu, xiang su, <lora:pixel book:1>'), 
        ('Pixel Bottle', 'bottle,pixel, pixel art, pixelart, xiangsu, xiang su, simple background, <lora:Pixel bottle:1>'), 
        ('Pixel Isometry', '((Isometry)), pixel, pixel art, solo, <lora:Pixel_Building2:1>'), 
        ]

theme = gr.themes.Monochrome()

#################################################
#################### Txt2Img ####################
#################################################

with gr.Blocks() as text_to_img:
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                show_label=True,
                max_lines=2,
                placeholder="Enter positive prompt"
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                show_label=True,
                max_lines=2,
                placeholder="Enter Negative prompt"
            )
            step_slider = gr.Slider(
                value=20,
                minimum=1,
                maximum=100,
                label="Step",
                show_label=True, 
                step=1
            )
            width_slider = gr.Slider(
                value=512,
                minimum=256,
                maximum=2048,
                label="Width",
                show_label=True, 
                step=64
            )
            height_slider = gr.Slider(
                value=512, 
                minimum=256, 
                maximum=2048, 
                label="Height", 
                show_label=True, 
                step=64
            )
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=model_names, 
                    value="anyloraCheckpoint_bakedvaeBlessedFp16.safetensors [5353d90e0c]", 
                    label="Select an Model", 
                    show_label=True, 
                    scale=5, 
                )
                model_open_button = gr.Button(
                    value="Open Model Folder", 
                    interactive=True, 
                )
                model_open_button.click(
                    fn=open_folder, 
                    inputs=[], 
                    outputs=[]
                    )
            lora_dropdown = gr.Dropdown(
                # choices=lora_list,
                choices=lora_names,
                # value='', 
                label="Select an LoRA",
                show_label=True, 
            )
            lora_dropdown.change(
            fn=add_loras,
            inputs=lora_dropdown, 
            outputs=prompt, 
            )
        with gr.Column():
            generate_button = gr.Button("Generate Image")
            t2i_result = gr.Image()

        generate_button.click(
            fn=txt2img.generate_image,
            inputs=[prompt, negative_prompt, step_slider, width_slider, height_slider, model_dropdown, lora_dropdown],
            outputs=t2i_result
            )
        

#################################################
#################### Img2Img ####################
#################################################

with gr.Blocks() as img_to_img:
    with gr.Row():
        with gr.Column():
            i2i_input = gr.Image(show_label=False)
            i2i_prompt = gr.Textbox(
                label="Prompt",
                show_label=True,
                max_lines=2,
                placeholder="Enter positive prompt", 
            )
            i2i_negative_prompt = gr.Textbox(
                label="Negative Prompt",
                show_label=True,
                max_lines=2,
                placeholder="Enter Negative prompt", 
            )
            i2i_step_slider = gr.Slider(
                value=20,
                minimum=1,
                maximum=100,
                label="Step",
                show_label=True, 
                step=1
            )
            i2i_width_slider = gr.Slider(
                value=512,
                minimum=256,
                maximum=2048,
                label="Width",
                show_label=True, 
                step=64
            )
            i2i_height_slider = gr.Slider(
                value=512,
                minimum=256,
                maximum=2048,
                label="Height",
                show_label=True, 
                step=64
            )
            denoising_strength_silder = gr.Slider(
                value=0.6,
                minimum=0,
                maximum=1,
                label="Denoising Strength",
                show_label=True, 
                step=0.05
            )
            with gr.Row():
                i2i_model_dropdown = gr.Dropdown(
                    choices=model_names,
                    # value="v1-5-pruned-emaonly.safetensors [6ce0161689]", 
                    value="anyloraCheckpoint_bakedvaeBlessedFp16.safetensors [5353d90e0c]", 
                    label="Select an Model", 
                    show_label=True, 
                    scale=4, 
                )
                model_open_button = gr.Button(
                    value="Open Model Folder", 
                    interactive=True, 
                    scale=1, 
                )
                model_open_button.click(fn=open_folder, inputs=[], outputs=[])
            i2i_lora_dropdown = gr.Dropdown(
                # choices=lora_list,
                choices=lora_names,
                # value='', 
                label="Select an LoRA",
                show_label=True, 
            )
            i2i_lora_dropdown.change(
            fn=add_loras,
            inputs=i2i_lora_dropdown, 
            outputs=i2i_prompt, 
            )
        with gr.Column():
            generate_button = gr.Button("Generate Image")
            i2i_result = gr.Image()

        generate_button.click(
            fn=img2img.generate_img2img,
            inputs=[i2i_input, i2i_prompt, i2i_negative_prompt, i2i_step_slider, i2i_width_slider, i2i_height_slider, denoising_strength_silder, i2i_model_dropdown, i2i_lora_dropdown],
            outputs=i2i_result
            )
        
#################################################
#################### Inpaint ####################
#################################################
        
################################################
################# Image Viewer #################
################################################

with gr.Blocks() as img_viewer_tab:
    with gr.Row():
        with gr.Column():
            t2i_folder_dropdown = gr.Dropdown(
                label="txt2img Folder Path",
                choices=img_viewer.get_folders_in_directory("stable-diffusion-webui/output/txt2img-images"),
            )
            i2i_folder_dropdown = gr.Dropdown(
                label="img2img Folder Path",
                choices=img_viewer.get_folders_in_directory("stable-diffusion-webui/output/img2img-images"),
            )
            bgremoved_folder_dropdown = gr.Dropdown(
                label="Background removed Folder Path",
                choices=img_viewer.get_folders_in_directory("background-removed-images"),
            )
            with gr.Row():
                t2i_refresh_button = gr.Button("txt2img Images Refresh")
                i2i_refresh_button = gr.Button("img2img Images Refresh")
                bgremoved_refresh_button = gr.Button("Background removed Images Refresh")
            button = gr.Button(
                value="Open Image Folder", 
                interactive=True, 
            )
            button.click(fn=open_image_folder, inputs=[], outputs=[])
        with gr.Column():
            img_view_result = gr.Gallery(label="Images", interactive=False)

        t2i_folder_dropdown.change(
            fn=img_viewer.load_images_from_folder,
            inputs=t2i_folder_dropdown, 
            outputs=img_view_result, 
        )
        i2i_folder_dropdown.change(
            fn=img_viewer.load_i2i_images_from_folder,
            inputs=i2i_folder_dropdown, 
            outputs=img_view_result, 
        )
        bgremoved_folder_dropdown.change(
            fn=img_viewer.load_bgremoved_images_from_folder,
            inputs=bgremoved_folder_dropdown, 
            outputs=img_view_result, 
        )
        t2i_refresh_button.click(
            fn=img_viewer.load_images_from_folder,
            inputs=t2i_folder_dropdown, 
            outputs=img_view_result
        )
        i2i_refresh_button.click(
            fn=img_viewer.load_i2i_images_from_folder,
            inputs=i2i_folder_dropdown, 
            outputs=img_view_result
        )
        bgremoved_refresh_button.click(
            fn=img_viewer.load_bgremoved_images_from_folder,
            inputs=bgremoved_folder_dropdown, 
            outputs=img_view_result
        )
        
################################################
############## Background Remover ##############
################################################

with gr.Blocks() as background_remover_tab:
    with gr.Row():
        with gr.Column():
            yolo_image = gr.Image(
                type="filepath", 
                label="Upload Image"
                )
            button = gr.Button(
                value="Open Background removed Folder", 
                interactive=True, 
            )
            button.click(fn=open_removed_folder, inputs=[], outputs=[])
        with gr.Column():
            remove_button = gr.Button("Start")
            removed_image = [gr.Image(
                label="Image with Bounding Boxes"
            ), 
            gr.Image(
                label="Cropped & Background Removed Image"
            )
            ]
        remove_button.click(
                fn=background_remover.background_remover_and_bbox,
                inputs=yolo_image,
                outputs=removed_image,
            )

#################################################
################### Interface ###################
#################################################

demo = gr.TabbedInterface(
    [text_to_img, img_to_img, background_remover_tab, img_viewer_tab], ["txt2img", "img2img", "Background Remover", "Image Viewer"], 
    title="Asset Generator",
    theme=theme
)

demo.launch()