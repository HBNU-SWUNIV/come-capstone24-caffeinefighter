import api_client
from PIL import Image

def generate_img2img(i2i_input, prompt, negative_prompt, applied_lora, step_slider, width_slider, height_slider, denoising_strength, model_dropdown, lora_dropdown):
    # prompt += ", " + lora_dropdown
    prompt += applied_lora
    negative_prompt += ", nsfw"
    api_client.change_model(model_dropdown)
    result2 = api_client.api.img2img(
        images=[Image.fromarray(i2i_input)], 
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