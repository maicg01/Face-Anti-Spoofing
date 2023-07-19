from __future__ import division
import datetime
import math
import numpy as np
import cv2
import torch
import torchvision


# take value of box and keypoits on face
def take_box_detector(img, detector):
    for _ in range(1):
        ta = datetime.datetime.now()
        bboxes, kpss = detector.detect(img, 0.7)
        tb = datetime.datetime.now()
        return bboxes, kpss

# alignment face
def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def alignment(img, l_eye, r_eye):
    left_eye_x = l_eye[0]; left_eye_y = l_eye[1]
    right_eye_x = r_eye[0]; right_eye_y = r_eye[1]

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
        print("rotate to clock direction")
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock
        # print("rotate to inverse clock direction")

    a = euclidean_distance(l_eye, point_3rd)
    b = euclidean_distance(r_eye, l_eye)
    c = euclidean_distance(r_eye, point_3rd)

    cos_a = (b*b + c*c - a*a)/(2*b*c)
    # print("cos(a) = ", cos_a)
    
    angle = np.arccos(cos_a)
    # print("angle: ", angle," in radian")
    
    angle = (angle * 180) / math.pi
    # print("angle: ", angle," in degree")

    if direction == -1:
        angle = 90 - angle

    from PIL import Image
    new_img = Image.fromarray(img)
    new_img = np.array(new_img.rotate(direction * angle))
    return new_img

# process keypoits to take distance between keypoits on face
def process_kps(kps):
    l_eye = kps[0]
    r_eye = kps[1]
    nose = kps[2]
    l_mouth = kps[3]
    r_mouth = kps[4]
    # kp = l4.astype(np.int)
    # cv2.circle(img, tuple(kp) , 1, (0,0,255) , 2)
    center1 = (l_eye + l_mouth)/2
    # print('=======================center1',center1)
    center2 = (r_eye + r_mouth)/2
    distance12 = math.dist(center1,center2)
    # print('=======================distance12',distance12)
    
    distance_nose1 = math.dist(center1, nose)
    distance_nose2 = math.dist(center2, nose)

    center_eye = (l_eye + r_eye)/2
    center_mouth = (l_mouth + r_mouth)/2
    distance_center_eye_mouth =  math.dist(center_eye,center_mouth)
    distance_nose_ceye = math.dist(center_eye, nose)
    distance_nose_cmouth = math.dist(center_mouth, nose)

    distance_eye = math.dist(l_eye,r_eye)
    distance_mouth = math.dist(l_eye,r_eye)

    return distance12, distance_nose1, distance_nose2, distance_center_eye_mouth, distance_nose_ceye, distance_nose_cmouth, distance_eye, distance_mouth, l_eye, r_eye

# process input image to return quality image and embbeding image
def process_onnx(img, session1, session2):
    # session1 model resnet: 2 output feature vecto quality
    # session2 model quality output quality image

    # normalized image
    img = cv2.resize(img, (112, 112))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        ccropped = img.swapaxes(1, 2).swapaxes(0, 1)
    except:
        print('error')
        return None
    ccropped = (ccropped - 127.5) / 128.0
    ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype = np.float32)
    

    results1 = session1.run(['output_0', 'output_1'], {'input': ccropped}) #input phai la mot array, kp torch
    print('results output', len(results1))
    
    feature = results1[0]
    quality = results1[1]

    results2 = session2.run(['output'], {'input': quality})

    rs_quality = results2[0]

    feature = torch.from_numpy(feature)
    print("shape cua feature: ", type(feature))

    return rs_quality[0], feature

def to_input(pil_rgb_image):
    np_img = cv2.resize(pil_rgb_image,(112,112))
    np_img = ((np_img / 255.) - 0.5) / 0.5
    try:
        np_img = np_img.swapaxes(1, 2).swapaxes(0, 1)
    except:
        print('error')
        return None
    np_img = np.reshape(np_img, [1, 3, 112, 112])
    np_img = np.array(np_img, dtype = np.float32)
    # tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return np_img

def process_onnx_adaFace(img,session):
    # bgr_tensor_input1 = cv2.imread(path)
    bgr_tensor_input1 = to_input(img)
    results = session.run(['output'], {'input': bgr_tensor_input1})
    feature = torch.from_numpy(results[0])
    return feature
