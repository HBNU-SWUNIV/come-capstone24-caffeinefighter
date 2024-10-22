import gradio as gr
import txt2img
import img2img
import inpaint
import os
import img_viewer
import background_remover_yolo
import background_remover_florence
import api_client

from googletrans import Translator

# 프롬프트에 로라 추가 시 필요
def add_loras(lora_name):
    lora = "<lora:" + lora_name + ":1>"
    return lora

# 파일 관리자에서 폴더 열기
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

lorafolder_path = "stable-diffusion-webui\models\Lora"
extensions = (".ckpt", ".safetensors")

#모델들, 현재 모델, 로라 이름 받아오기
model_names = api_client.api.util_get_model_names()
current_model = api_client.api.util_get_current_model()
lora_names = api_client.api.util_get_lora_names()

language_options = {
    'auto': 'Auto Detection',
    'zh-cn': 'Chinese(Simplified)',
    'zh-tw': 'Chinese(traditional)',
    'en': 'English',
    'fr': 'French',
    'de': 'German',
    'ja': 'Japanese',
    'ko': 'Korean',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'es': 'Spanish',
    'tr': 'Turkish'
}

theme = gr.themes.Monochrome()

#################################################
################### Translate ###################
#################################################

translator = Translator()

def translate_prompt(prompt,lang):
    try:
        translated = translator.translate(prompt, src= lang, dest='en')
        return translated.text
    except Exception as e:
        return f"번역 오류: {e}"

#################################################
#################### Txt2Img ####################
#################################################

with gr.Blocks() as text_to_img:
    with gr.Row():
        with gr.Column():
            lang_dropdown = gr.Dropdown(
                choices=[(name, code) for code, name in language_options.items()],
                label="Input Language",
                value="auto", 
                show_label=True,
            )
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
            applied_lora = gr.Textbox(
                label="Applied LoRA",
                show_label=True,
                max_lines=2,
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
            outputs=applied_lora, 
            )
        with gr.Column():
            generate_button = gr.Button("Generate Image")
            t2i_result = gr.Image()
            translated_positive_prompt = gr.Textbox(
                label="Translated Positive Prompt",
                show_label=True,
                interactive=False
            )
            translated_negative_prompt = gr.Textbox(
                label="Translated Negative Prompt",
                show_label=True,
                interactive=False
            )

        # 번 역 부 분 #
        def generate_image_with_translation(prompt, negative_prompt, applied_lora, steps, width, height, model_name, lora_name, lang):
            prompt_en = translate_prompt(prompt, lang) 
            negative_prompt_en = translate_prompt(negative_prompt,lang)  
            return txt2img.generate_image(prompt_en, negative_prompt_en, applied_lora, steps, width, height, model_name, lora_name), prompt_en, negative_prompt_en

        generate_button.click(
            fn=generate_image_with_translation,
            #fn=txt2img.generate_image,
            inputs=[prompt, negative_prompt, applied_lora, step_slider, width_slider, height_slider, model_dropdown, lora_dropdown, lang_dropdown],
            outputs=[t2i_result, translated_positive_prompt, translated_negative_prompt]
            )


#################################################
#################### Img2Img ####################
#################################################

with gr.Blocks() as img_to_img:
    with gr.Row():
        with gr.Column():
            i2i_input = gr.Image(show_label=False)
            lang_dropdown = gr.Dropdown(
                choices=[(name, code) for code, name in language_options.items()],
                label="Input Language",
                value="auto", 
                show_label=True,
            )
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
            i2i_applied_lora = gr.Textbox(
                label="Applied LoRA",
                show_label=True,
                max_lines=2,
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
            i2i_denoising_strength_silder = gr.Slider(
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
            outputs=i2i_applied_lora, 
            )
        with gr.Column():
            generate_button = gr.Button("Generate Image")
            i2i_result = gr.Image()
            translated_positive_prompt = gr.Textbox(
                label="Translated Positive Prompt",
                show_label=True,
                interactive=False
            )
            translated_negative_prompt = gr.Textbox(
                label="Translated Negative Prompt",
                show_label=True,
                interactive=False
            )

        def generate_image_with_translation(i2i_input, i2i_prompt, i2i_negative_prompt, i2i_applied_lora, steps, width, height, denoising_strength, model_name, lora_name, lang):
            i2i_prompt_en = translate_prompt(i2i_prompt, lang) 
            i2i_negative_prompt_en = translate_prompt(i2i_negative_prompt, lang)  
            return img2img.generate_img2img(i2i_input, i2i_prompt_en, i2i_negative_prompt_en, i2i_applied_lora, steps, width, height, denoising_strength, model_name, lora_name), i2i_prompt_en, i2i_negative_prompt_en
        
        generate_button.click(
            fn=generate_image_with_translation,
            #fn=img2img.generate_img2img,
            inputs=[i2i_input, i2i_prompt, i2i_negative_prompt, i2i_applied_lora, i2i_step_slider, i2i_width_slider, i2i_height_slider, i2i_denoising_strength_silder, i2i_model_dropdown, i2i_lora_dropdown, lang_dropdown],
            outputs=[i2i_result, translated_positive_prompt, translated_negative_prompt]
            )
        
#################################################
#################### Inpaint ####################
#################################################

with gr.Blocks() as inpaint_tab:
    with gr.Row():
        with gr.Column():
            in_mask = gr.ImageMask(
                label="Inpaint", 
                show_label=True, 
            )
            lang_dropdown = gr.Dropdown(
                choices=[(name, code) for code, name in language_options.items()],
                label="Input Language",
                value="auto", 
                show_label=True,
            )
            in_prompt = gr.Textbox(
                label="Prompt",
                show_label=True,
                max_lines=2,
                placeholder="Enter positive prompt", 
            )
            in_negative_prompt = gr.Textbox(
                label="Negative Prompt",
                show_label=True,
                max_lines=2,
                placeholder="Enter Negative prompt", 
            )
            in_applied_lora = gr.Textbox(
                label="Applied LoRA",
                show_label=True,
                max_lines=2,
            )
            in_step_slider = gr.Slider(
                value=20,
                minimum=1,
                maximum=100,
                label="Step",
                show_label=True, 
                step=1
            )
            in_width_slider = gr.Slider(
                value=512,
                minimum=256,
                maximum=2048,
                label="Width",
                show_label=True, 
                step=64
            )
            in_height_slider = gr.Slider(
                value=512,
                minimum=256,
                maximum=2048,
                label="Height",
                show_label=True, 
                step=64
            )
            in_denoising_strength_silder = gr.Slider(
                value=0.6,
                minimum=0,
                maximum=1,
                label="Denoising Strength",
                show_label=True, 
                step=0.05
            )
            with gr.Row():
                in_model_dropdown = gr.Dropdown(
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
            in_lora_dropdown = gr.Dropdown(
                # choices=lora_list,
                choices=lora_names,
                # value='', 
                label="Select an LoRA",
                show_label=True, 
            )
            in_lora_dropdown.change(
            fn=add_loras,
            inputs=in_lora_dropdown, 
            outputs=in_applied_lora, 
            )
        with gr.Column():
            generate_button = gr.Button("Generate Image")
            in_result = gr.Image()
            translated_positive_prompt = gr.Textbox(
                label="Translated Positive Prompt",
                show_label=True,
                interactive=False
            )
            translated_negative_prompt = gr.Textbox(
                label="Translated Negative Prompt",
                show_label=True,
                interactive=False
            )

        def generate_image_with_translation(in_mask,in_prompt, in_negative_prompt, in_applied_lora, steps, width, height, denoising_strength, model_name, lora_name, lang):
            in_prompt_en = translate_prompt(in_prompt, lang) 
            in_negative_prompt_en = translate_prompt(in_negative_prompt, lang)  
            return inpaint.generate_inpaint(in_mask, in_prompt_en, in_negative_prompt_en, in_applied_lora, steps, width, height, denoising_strength, model_name, lora_name), in_prompt_en, in_negative_prompt_en
        
        generate_button.click(
            fn=generate_image_with_translation, 
            inputs=[in_mask, in_prompt, in_negative_prompt, in_applied_lora, in_step_slider, in_width_slider, in_height_slider, in_denoising_strength_silder, in_model_dropdown, in_lora_dropdown, lang_dropdown],
            outputs=[in_result, translated_positive_prompt, translated_negative_prompt]
            )

################################################
################# Image Viewer #################
################################################

with gr.Blocks() as img_viewer_tab:
    with gr.Row():
        with gr.Column():
            # def update_choices():
            #     choices=img_viewer.get_folders_in_directory("stable-diffusion-webui/output/txt2img-images")
            #     return gr.Dropdown(choices=choices, interactive=True)=
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
        # t2i_refresh_button.click(
        #     fn=img_viewer.load_images_from_folder,
        #     outputs=t2i_folder_dropdown
        # )
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
            yolo_image = gr.File(
                type="filepath", 
                label="Upload Image",
                file_count='multiple',
                )
            button = gr.Button(
                value="Open Background removed Folder", 
                interactive=True, 
            )
            button.click(fn=open_removed_folder, inputs=[], outputs=[])
        with gr.Column():
            remove_button = gr.Button("Start")
            removed_image = [
                gr.Gallery(label="Image with Bounding Boxes"), 
                gr.Gallery(label="Cropped & Background Removed Image")
            ]
        remove_button.click(
                # fn=background_remover_yolo.background_remover_and_bbox,
                fn=background_remover_florence.background_remover_and_bbox,
                inputs=yolo_image,
                outputs=removed_image,
            )

#################################################
################### Interface ###################
#################################################

demo = gr.TabbedInterface(
    [text_to_img, img_to_img, inpaint_tab, background_remover_tab, img_viewer_tab], ["txt2img", "img2img", "Inpaint", "Background Remover", "Image Viewer"], 
    title="Asset Generator",
    theme=theme
)

demo.launch()