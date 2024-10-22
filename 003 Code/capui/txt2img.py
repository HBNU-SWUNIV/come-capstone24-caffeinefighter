import api_client

import torch
from PIL import Image
import numpy as np

model = torch.hub.load("yolov5", 'custom', "yolov5\yolov5s.pt", source='local')

def generate_image(prompt, negative_prompt, applied_lora, step_slider, width_slider, height_slider, model_dropdown, lora_dropdown):
    prompt += applied_lora
    negative_prompt += ", nsfw"
    api_client.change_model(model_dropdown)
    result = api_client.api.txt2img(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        seed=-1, 
        cfg_scale=7, 
        sampler_index="Euler a", 
        steps=step_slider, 
        width=width_slider, 
        height=height_slider, 
        save_images=True
    )

    return result.image

# def generate_image(prompt, negative_prompt, seed, cfg_scale, sampler_index, steps, height, width, save_images):
#     result = api_client.api.txt2img(
#         prompt=prompt,
#         negative_prompt=negative_prompt,
#         seed=seed, 
#         cfg_scale=cfg_scale, 
#         sampler_index=sampler_index, 
#         steps=steps, 
#         height=height, 
#         width=width, 
#         save_images=save_images
#     )
#     return result

# # images contains the returned images (PIL images)
# result1.images

# # image is shorthand for images[0]
# result1.image

# # info contains text info about the api call
# result1.info

# # info contains paramteres of the api call
# result1.parameters

# result1.image