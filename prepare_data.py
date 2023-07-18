import cv2
import numpy as np
import insightface

import os
import argparse
import warnings
import time
from src.generate_patches import CropImage
from src.utility import parse_model_name


def take_image(image):
    height, width = image.shape[:2]
    if height / width != 4 / 3:
        if width > height:
            new_width = height * 3 // 4

            # Tính toán vị trí cắt
            x_offset = (width - new_width) // 2
            if x_offset < 0:
                x_offset = 0

            # Cắt ảnh
            cropped_image = image[:, x_offset:x_offset+new_width, :]
            cv2.imwrite('new_image.jpg', cropped_image)
            toado = [0, x_offset]
            return cropped_image, toado
        else:
            new_height = width * 4 // 3

            # Tính toán vị trí cắt
            y_offset = (height - new_height) // 2
            if y_offset < 0:
                y_offset = 0
            cropped_image = image[y_offset:y_offset+new_height, :, :]
            cv2.imwrite('new_image.jpg', cropped_image)
            toado = [y_offset, 0]
            return cropped_image, toado

def crop_image(image, bbox):
    x, y, x2, y2 = bbox
    height, width = image.shape[:2]
    d_org_x2 = width - x2
    d_org_y2 = height - y2
    print(d_org_x2)

    if x<d_org_x2:
        x_start = 0
        x_end = x2 + x
    else:
        x_start = x - d_org_x2
        x_end = width

    if y < d_org_y2:
        y_start = 0
        y_end = y2 + y
    else:
        y_start = y - d_org_y2
        y_end = height


    crop_img = image[y_start:y_end, x_start:x_end]

    new_bbox_x = x - x_start
    new_bbox_y = y - y_start
    new_bbox_width = x2 -x
    new_bbox_height = y2 -y

    image_crop_final, toado = take_image(crop_img)
    new_bbox_x = new_bbox_x - toado[1]
    new_bbox_y = new_bbox_y - toado[0]

    image_bbox_final = [new_bbox_x, new_bbox_y, new_bbox_width, new_bbox_height]

    return image_crop_final, image_bbox_final

def prepare_data(image, bbox, w_input, h_input, dir_save, name_image, name_real_fake, scale=None):
    image_cropper = CropImage()
    image_crop, image_bbox = crop_image(image, bbox)
    param = {
        "org_img": image_crop,
        "bbox": image_bbox,
        "scale": scale,
        "out_w": w_input,
        "out_h": h_input,
        "crop": True,
    }
    if scale is None:
        param["crop"] = False
    img = image_cropper.crop(**param)
    if scale == None:
        dir_save_folder = dir_save + '/org_1_80x60/' + str(name_real_fake) 
    else:
        dir_save_folder = dir_save + '/' +  str(scale) + '_80x80/' + str(name_real_fake) 

    if not os.path.exists(dir_save_folder):
        os.makedirs(dir_save_folder)
        print(f"Created folder '{dir_save_folder}'")
    else:
        print(f"Folder '{dir_save_folder}' already exists")

    path_save_image = dir_save_folder + '/' + str(name_image) + '.png'
    cv2.imwrite(path_save_image, img)

if __name__ == "__main__":
    dir_save = '/home/maicg/Documents/Me/ANTI-FACE/Face-Anti-Spoofing/datasets/rgb_image'
    dir_data = ''
    