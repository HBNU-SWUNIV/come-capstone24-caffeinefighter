import api_client
from PIL import Image

def generate_inpaint(in_input, mask_image, prompt, negative_prompt, step_slider, width_slider, height_slider, denoising_strength, model_dropdown, lora_dropdown):
    prompt += ", " + lora_dropdown
    api_client.change_model(model_dropdown)
    result2 = api_client.api.img2img(
        images=[Image.fromarray(in_input)], 
        mask_image=mask_image,
        inpainting_fill=1,
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