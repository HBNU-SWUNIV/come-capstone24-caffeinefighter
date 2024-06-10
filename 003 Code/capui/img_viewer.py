import os
from PIL import Image

# 기본 경로 설정
t2i_base_directory = "stable-diffusion-webui/output/txt2img-images"
i2i_base_directory = "stable-diffusion-webui/output/img2img-images"
bgremoved_base_directory = "background-removed-images"

def get_folders_in_directory(directory):
    full_path = os.path.join(os.getcwd(), directory)
    return [folder for folder in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, folder))]

def load_images_from_folder(folder_name):
    folder_path = os.path.join(t2i_base_directory, folder_name)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"지정된 경로를 찾을 수 없습니다: {folder_path}")
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            images.append(img)
    return images

def load_i2i_images_from_folder(folder_name):
    folder_path = os.path.join(i2i_base_directory, folder_name)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"지정된 경로를 찾을 수 없습니다: {folder_path}")
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            images.append(img)
    return images

def load_bgremoved_images_from_folder(folder_name):
    folder_path = os.path.join(bgremoved_base_directory, folder_name)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"지정된 경로를 찾을 수 없습니다: {folder_path}")
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            images.append(img)
    return images