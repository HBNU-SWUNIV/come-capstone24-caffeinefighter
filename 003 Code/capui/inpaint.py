import api_client
from PIL import Image
import numpy as np

def generate_inpaint(in_mask, prompt, negative_prompt, applied_lora, step_slider, width_slider, height_slider, denoising_strength, model_dropdown, lora_dropdown):
    image = in_mask['background']
    mask = in_mask['layers']
    com = in_mask['composite']

    prompt += applied_lora
    negative_prompt += ", nsfw"
    api_client.change_model(model_dropdown)
    result2 = api_client.api.img2img(
        images=[Image.fromarray(image)],
        mask_image=Image.fromarray(mask[0]),
        inpainting_fill=1,
        inpainting_mask_invert=0,
        inpaint_full_res=False,
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        seed=-1, 
        cfg_scale=7, 
        sampler_index="Euler a", 
        steps=step_slider, 
        width=width_slider, 
        height=height_slider, 
        denoising_strength=denoising_strength, 
        save_images=True
    )

    return result2.image