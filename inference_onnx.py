import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

import os
import argparse
import warnings
import time

import onnxruntime
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


path = "/home/maicg/Downloads/photo_2023-03-10_17-39-16" #khong co .jpg

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image, dtype = np.float32)
    # np_img = ((np_img / 255.) - 0.5) / 0.5
    # np_img = np_img / 255.
    try:
        np_img = np_img.swapaxes(1, 2).swapaxes(0, 1)
    except:
        print('error')
        return None
    np_img = np.reshape(np_img, [1, 3, 80, 80])
    
    # tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return np_img

def load_model_onnx(file_name):
    session = onnxruntime.InferenceSession(file_name, providers=['CUDAExecutionProvider'])
    return session

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

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
img = ins_get_image(path)
faces = app.get(img)
rimg = app.draw_on(img, faces)
cv2.imwrite("./t1_output.jpg", rimg)

for face in faces:
    bbox = face.bbox.astype(np.int)
    x, y, x2, y2 = bbox
    cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(x, y, x2, y2)

image = cv2.imread(path + ".jpg")
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

cv2.imshow('crop', crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

process_image, toado = take_image(crop_img)
new_bbox_x = new_bbox_x - toado[1]
new_bbox_y = new_bbox_y - toado[0]

cv2.imshow('process_crop', process_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def test(image, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    # image_bbox = model_test.get_bbox(image)
    # print("image_bbox: ", image_bbox)
    image_bbox = [new_bbox_x, new_bbox_y, new_bbox_width, new_bbox_height]
    print("image_bbox: ", image_bbox)
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        path_model = os.path.join(model_dir,model_name)
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        print("shape: ", img.shape)
        input = to_input(img)
        start = time.time()
        session = load_model_onnx(path_model)
        results = session.run(['output'], {'input': input})
        rs = results[0]
        softmax_x = np.exp(rs) / np.sum(np.exp(rs), axis=1, keepdims=True)
        print(softmax_x)
        prediction += softmax_x
        test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    if label == 1:
        print("Image '{}' is Real Face. Score: {:.2f}.".format("image_name", value))
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        print("Image '{}' is Fake Face. Score: {:.2f}.".format("image_name", value))
        result_text = "FakeFace Score: {:.2f}".format(value)
        color = (0, 0, 255)
    print("Prediction cost {:.2f} s".format(test_speed))
    cv2.rectangle(
        image,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        image,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)
    
    cv2.imwrite("results.jpg", image)


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./onnx",
        help="model_lib used to test")
    parser.add_argument(
        "--image_name",
        type=str,
        default="image_F1.jpg",
        help="image used to test")
    args = parser.parse_args()
    test(process_image, args.model_dir, args.device_id)


