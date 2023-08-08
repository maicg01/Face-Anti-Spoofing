from __future__ import division
import torch
# from models.model_resnet import ResNet, FaceQuality
from utils.convert_onnx import load_state_dict, convert_ONNX_1output, convert_ONNX_adaFace
from src.anti_spoof_predict import AntiSpoofPredict
from src.model_lib.MiniFASNet import MiniFASNetV2


def main():
    model = AntiSpoofPredict(0)
    model = model._load_model("/home/maicg/Documents/Me/ANTI-FACE/Face-Anti-Spoofing/resources/new/2.7_80x80_MiniFASNetV2.pth")

    # #convert to onnx
    # # Let's load the model we just created and test the accuracy per label 
    # model = ResNet(num_layers=100, feature_dim=512)
    # path_resnet = './weights/backbone.pth'
    # load_state_dict(model, torch.load(path_resnet))
    model.eval()
    print("doneeeee")
 
    # Conversion to ONNX 
    convert_ONNX_adaFace(model) 

main()