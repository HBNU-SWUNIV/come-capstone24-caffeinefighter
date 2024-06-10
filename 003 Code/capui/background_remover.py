import torch
from PIL import Image
from rembg import remove
import os
from datetime import datetime

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

bgremoved_images_directory = "background-removed-images"

def background_remover_and_bbox(image, padding=30):
    img = Image.open(image)

    base_name = os.path.basename(image)  # 이미지 원본 이름 추출
    name, ext = os.path.splitext(base_name)
    new_name = f"{name}_removed.png"

    current_date = datetime.now().strftime("%Y-%m-%d")
    folder_path = os.path.join(bgremoved_images_directory, current_date)
    os.makedirs(folder_path, exist_ok=True)
    
    # 파일 이름 중복 확인 및 새로운 이름 생성
    image_path = os.path.join(folder_path, new_name)
    unique_suffix = 1
    while os.path.exists(image_path):  # 이미 존재하는 파일 이름인 경우
        new_name = f"{name}_removed_{unique_suffix}.png"
        image_path = os.path.join(folder_path, new_name)
        unique_suffix += 1

    results = model(img)
    results.render()  # bounding box 그리기
    img_with_boxes = Image.fromarray(results.ims[0])  # bounding box가 그려진 이미지

    for *box, conf, cls in results.xyxy[0]:  # 첫 번째 이미지의 모든 bbox
        x1, y1, x2, y2 = map(int, box)
        # 여백을 추가하여 좌표 조정
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img.width, x2 + padding)
        y2 = min(img.height, y2 + padding)
        cropped_img = img.crop((x1, y1, x2, y2))
        background_removed_img = remove(cropped_img)  # 크롭 이미지에서 배경제거
        background_removed_img.save(image_path)  # 이미지 저장
        return img_with_boxes, background_removed_img

# def background_remover_and_bbox(image,padding=30):
#     img = Image.open(image)

#     base_name = os.path.basename(image) # 이미지 원본 이름 추출
#     name, ext = os.path.splitext(base_name)
#     new_name = f"{name}_removed.png"

#     current_date = datetime.now().strftime("%Y-%m-%d")
#     folder_path = os.path.join(bgremoved_images_directory, current_date)
#     os.makedirs(folder_path, exist_ok=True)
#     image_path = os.path.join(folder_path, new_name)

#     results = model(img)
#     results.render()  # bounding box 그리기
#     img_with_boxes = Image.fromarray(results.ims[0])  # bounding box가 그려진 이미지

#     for *box, conf, cls in results.xyxy[0]:  # 첫 번째 이미지의 모든 bbox
#         x1, y1, x2, y2 = map(int, box)
#         # 여백을 추가하여 좌표 조정
#         x1 = max(0, x1 - padding)
#         y1 = max(0, y1 - padding)
#         x2 = min(img.width, x2 + padding)
#         y2 = min(img.height, y2 + padding)
#         cropped_img = img.crop((x1, y1, x2, y2))
#         background_removed_img = remove(cropped_img)  # 크롭 이미지에서 배경제거
#         background_removed_img.save(image_path)       # 이미지 저장
#         return img_with_boxes, background_removed_img  # cropped_img  # bounding box가 그려진 이미지와 첫 번째 bbox에 대한 크롭된 이미지 반환

########################################################
######## rembg 라이브러리 쓴 코드 (파라미터값조절)########
########################################################
# import torch
# from PIL import Image
# from rembg import remove

# # YOLOv5 모델 로드
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # def background_remove(image):
# #     img = Image.open(image)
# #     results = model(img)
# #     results.render()  # bounding box 그리기
# #     for *box, conf, cls in results.xyxy[0]:  # 첫 번째 이미지의 모든 bbox
# #         x1, y1, x2, y2 = map(int, box)
# #         cropped_img = img.crop((x1, y1, x2, y2))
# #         return cropped_img  # 첫 번째 bbox에 대한 크롭된 이미지 반환

# def background_remover_and_bbox(image):
#     img = Image.open(image)
#     results = model(img)
#     results.render()  # bounding box 그리기
#     img_with_boxes = Image.fromarray(results.ims[0])  # bounding box가 그려진 이미지
#     for *box, conf, cls in results.xyxy[0]:  # 첫 번째 이미지의 모든 bbox
#         x1, y1, x2, y2 = map(int, box)
#         cropped_img = img.crop((x1, y1, x2, y2))
#         parmas = {
#             "alpha_matting" : True,
#             "alpha_matting_foregorund_threshold" : 240,
#             "alpha_matting_background_threshold" : 15,
#             "alpha_matting_erode_structure_size" : 10,
#         }
#         background_removed_img = remove(cropped_img, **parmas)  # 크롭 이미지에서 배경제거

#         return img_with_boxes, background_removed_img  #cropped_img  # bounding box가 그려진 이미지와 첫 번째 bbox에 대한 크롭된 이미지 반환

