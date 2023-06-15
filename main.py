from pickletools import read_uint1
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.uic import loadUi
from butterworth import Butter
import sys

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = "./images/"


# 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        return True
    else:
        return True


def test(image_name, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = image_name
    result = check_image(image)
    if result is False:
        return
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
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
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    if label == 1:
        result_text = "REAL"
        color = (255, 0, 0)
    else:
        result_text = "FAKE"
        color = (0, 0, 255)
    print("Prediction cost {:.2f} s".format(test_speed))
    cv2.rectangle(
        image,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 4)
    cv2.putText(
        image,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX,3, color, 3)

    # face_name = image_name.split(".")[0]
    # final_name = face_name + "_result.jpg"
    # cv2.imwrite(final_name, image)
    return image

class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        loadUi("face_anti.ui", self)
        self.setWindowIcon(QtGui.QIcon("python-icon.png"))

        self.pre_img = None
        self.origin_img = None
        self.image = None

        #set input button
        # self.setQratio.setText(str(self.quantizationRatio))
        self.chooseImage.clicked.connect(self.open_img)
        self.start.clicked.connect(self.computeResult)
        self.Reset.clicked.connect(self.computeReset)

    @pyqtSlot()
    def loadImage(self, fname):
        self.image = cv2.imread(fname)
        self.origin_img = cv2.resize(self.image,(471,471))
        self.tmp = self.image
        self.displayImage()
    
    def displayImage(self, window=1):
        qformat = QImage.Format_Indexed8

        if len(self.origin_img.shape) == 3:
            if(self.origin_img.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.origin_img, self.origin_img.shape[1], self.origin_img.shape[0], self.origin_img.strides[0], qformat)
        # image.shape[0] là số pixel theo chiều Y
        # image.shape[1] là số pixel theo chiều X
        # image.shape[2] lưu số channel biểu thị mỗi pixel
        img = img.rgbSwapped() # chuyển đổi hiệu quả một ảnh RGB thành một ảnh BGR.
        if window == 1:
            self.pre_frame.setPixmap(QPixmap.fromImage(img))
            self.pre_frame.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)# căn chỉnh vị trí xuất hiện của hình trên lable
        if window == 2:
            self.aft_frame.setPixmap(QPixmap.fromImage(img))
            self.aft_frame.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def displayPreImage(self, window=1):
        qformat = QImage.Format_Indexed8

        if len(self.pre_img.shape) == 3:
            if(self.pre_img.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.pre_img, self.pre_img.shape[1], self.pre_img.shape[0], self.pre_img.strides[0], qformat)
        # image.shape[0] là số pixel theo chiều Y
        # image.shape[1] là số pixel theo chiều X
        # image.shape[2] lưu số channel biểu thị mỗi pixel
        img = img.rgbSwapped() # chuyển đổi hiệu quả một ảnh RGB thành một ảnh BGR.
        if window == 1:
            self.pre_frame.setPixmap(QPixmap.fromImage(img))
            self.pre_frame.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)# căn chỉnh vị trí xuất hiện của hình trên lable
        if window == 2:
            self.aft_frame.setPixmap(QPixmap.fromImage(img))
            self.aft_frame.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


    def open_img(self):
        fname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'This PC', "Image Files (*)")
        if fname:
            self.loadImage(fname)
        else:
            print("Invalid Image")  
    
    def computeResult(self):
        use_gpu = 1
        model = "./resources/anti_spoof_models"
        pre_img = test(self.image, model, use_gpu)
        self.pre_img = cv2.resize(pre_img,(471,471))
        self.displayPreImage(2)
        

    def computeReset(self):
        self.pre_frame.clear()
        self.aft_frame.clear()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = UI()
    win.show()
    sys.exit(app.exec())
