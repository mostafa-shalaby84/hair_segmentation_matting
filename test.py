import argparse
import torch
import numpy as np
from skimage import measure
import net
from deploy import inference_img_whole
from model import BiSeNet
import torchvision.transforms as transforms

import cv2
import os
from skimage.morphology import remove_small_holes, remove_small_objects
import sys

sys.path += os.path.abspath(__file__)

modelFile = "models/face_detector.pb"
configFile = "models/face_detector.pbtxt"
face_detector = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

n_classes = 19
seg_net = BiSeNet(n_classes=n_classes)
seg_net.load_state_dict(torch.load('models/segmentation.pth', map_location='cpu'))

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.resume = "models/matting.pth"
args.stage = 1
args.crop_or_resize = "whole"
args.max_size = 512
args.cuda = False
# init model
matting_model = net.VGG16(args)
ckpt = torch.load(args.resume, map_location='cpu')
matting_model.load_state_dict(ckpt['state_dict'], strict=True)


def segmentation(image):
    seg_net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        img = cv2.resize(image, (512, 512), cv2.INTER_LINEAR)
        img = to_tensor(img)
        img = torch.unsqueeze(img, 0)
        out = seg_net(img)[0]
        resize_out = np.zeros((n_classes, image.shape[0], image.shape[1]))
        for i in range(n_classes):
            resize_out[i] = cv2.resize(np.array(out[0][i]), (image.shape[1], image.shape[0]), cv2.INTER_CUBIC)
        parsing = resize_out.argmax(0)
        return parsing


def face_detect(image, conf_threshold=0.8):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)

    face_detector.setInput(blob)
    detections = face_detector.forward()

    if detections.shape[2] == 0:
        return None

    confidence = detections[0, 0, 0, 2]
    if confidence < conf_threshold:
        return
    x1 = int(detections[0, 0, 0, 3] * frameWidth)
    y1 = int(detections[0, 0, 0, 4] * frameHeight)
    x2 = int(detections[0, 0, 0, 5] * frameWidth)
    y2 = int(detections[0, 0, 0, 6] * frameHeight)

    return [x1, y1, x2, y2]


def face_crop(image, rect):
    height, width, _ = image.shape
    face_len = (rect[3] - rect[1]) * 2
    left = (rect[2] + rect[0]) // 2 - face_len // 2
    right = left + face_len
    top = rect[1] - face_len // 4
    bottom = top + face_len
    face_image = np.zeros((face_len, face_len, 3))
    out_left = max(0, -left)
    out_right = min(face_len, face_len - right + width)
    out_top = max(0, -top)
    out_bottom = min(face_len, face_len - bottom + height)
    in_left = max(0, left)
    in_right = min(width, right)
    in_top = max(0, top)
    in_bottom = min(height, bottom)
    face_image[out_top:out_bottom, out_left:out_right, :] = image[in_top:in_bottom, in_left:in_right, :]
    return face_image, [out_left, out_top, out_right, out_bottom], face_len


def pre_processing(seg_image):
    # get roi regions

    hair_image = seg_image == 17
    hat_image = seg_image == 18
    background = seg_image == 0

    HAT_REGION_THRESHOLD = 20000
    properties = measure.regionprops(hat_image.astype(np.uint8))
    if len(properties) > 0:
        if properties[0].area < HAT_REGION_THRESHOLD:
            background = background | hat_image
            hat_image = (np.zeros(hat_image.shape)).astype(np.bool)
        else:
            hat_image = remove_small_holes(hat_image, area_threshold=1000)
            hat_image = remove_small_objects(hat_image, min_size=HAT_REGION_THRESHOLD)

    hair_image = remove_small_holes(hair_image, area_threshold=1000)
    background = remove_small_holes(background, area_threshold=1000)
    background = remove_small_objects(background, min_size=10000)

    body_image = (np.ones((512, 512))).astype(np.bool)
    body_image = body_image ^ hair_image
    body_image = body_image ^ hat_image
    body_image = body_image ^ background

    roi_image = np.zeros((512, 512, 3))
    roi_image[:,:,0] = 255 * hair_image + body_image * 255
    roi_image[:, :, 1] = 255 * hair_image + body_image * 255
    roi_image[:, :, 2] = 255 * hair_image
    #cv2.imwrite("result/segmentation.png", roi_image)
    # get trimap image
    k_len = 24
    kernel = np.ones((k_len, k_len), np.uint8)
    enlarge_image = cv2.dilate(hair_image.astype(np.uint8), kernel)

    properties = measure.regionprops(hair_image.astype(np.uint8))
    if len(properties) == 0:
        return

    k_len = 4
    kernel = np.ones((k_len, k_len), np.uint8)
    threshold = properties[0].area // 2
    while properties[0].area > threshold:
        body_image = (cv2.dilate(body_image.astype(np.uint8), kernel)).astype(np.bool)
        body_image = (body_image & (~background) & (~hat_image))

        background = (cv2.dilate(background.astype(np.uint8), kernel)).astype(np.bool)
        background = (background & (~body_image) & (~hat_image))
        hair_image = hair_image & (~body_image) & (~background)
        properties = measure.regionprops(hair_image.astype(np.uint8))

    trip_image = enlarge_image * 127 + hair_image * 128
    #cv2.imwrite("result/trimap.png", trip_image)
    return background, body_image, hat_image, hair_image, trip_image


def processing(image_path, file_title):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_rect = face_detect(image)
    if face_rect is None:
        return

    frameOpencvDnn = image.copy()
    cv2.rectangle(frameOpencvDnn, (face_rect[0], face_rect[1]), (face_rect[2], face_rect[3]), (0, 255, 0), int(round(image.shape[0] / 150)), 8)
    #cv2.imwrite("result/detect.png", cv2.cvtColor(frameOpencvDnn, cv2.COLOR_BGR2RGB))

    face_image, image_rect, image_len = face_crop(image, face_rect)
    resize_image = cv2.resize(face_image, (512, 512), cv2.INTER_LINEAR).astype(np.uint8)

    #cv2.imwrite("result/normalize.png", cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB))

    seg_image = segmentation(resize_image)

    background, body_image, hat_image, hair_image, trip_image = pre_processing(seg_image)

    matting_image = inference_img_whole(args, matting_model, resize_image, trip_image)

    #cv2.imwrite("result/matting.png", matting_image * 255)

    alpha_image = np.zeros(resize_image.shape)
    alpha_image[:, :, 0] = matting_image * 255 + hat_image * 255 * (1 - matting_image) + hair_image * 255 * (
                1 - matting_image)
    alpha_image[:, :, 1] = matting_image * 255 + body_image * 255 * (1 - matting_image) + hair_image * 255 * (
                1 - matting_image)
    alpha_image[:, :, 2] = matting_image * 255 + body_image * 255 * (1 - matting_image) + hair_image * 255 * (
                1 - matting_image)
    bk_image = (resize_image + alpha_image) * 0.5

    alpha_image = cv2.cvtColor(alpha_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    resize_image = cv2.cvtColor(resize_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    bk_image = cv2.cvtColor(bk_image.astype(np.uint8), cv2.COLOR_BGR2RGB)

    alpha_image = cv2.resize(alpha_image, (image_len, image_len), cv2.INTER_LINEAR)
    resize_image = cv2.resize(resize_image, (image_len, image_len), cv2.INTER_LINEAR)
    bk_image = cv2.resize(bk_image, (image_len, image_len), cv2.INTER_LINEAR)

    alpha_image = alpha_image[image_rect[1]: image_rect[3], image_rect[0]: image_rect[2]]
    resize_image = resize_image[image_rect[1]: image_rect[3], image_rect[0]: image_rect[2]]
    bk_image = bk_image[image_rect[1]: image_rect[3], image_rect[0]: image_rect[2]]

    cv2.imwrite("result/{}_!real.png".format(file_title), resize_image)
    cv2.imwrite("result/{}_alpha.png".format(file_title), alpha_image)
    cv2.imwrite("result/{}_composition.png".format(file_title), bk_image)

    wid = 600
    hei = wid * alpha_image.shape[0] // alpha_image.shape[1]
    cv2.imshow('alpha_image', cv2.resize(alpha_image, (wid, hei)))
    cv2.imshow('bk_image', cv2.resize(bk_image, (wid, hei)))
    cv2.waitKey(0)


#processing("images/color_1.jpg", "color_1")
image_dir = "images"
for file in os.listdir(image_dir):
    if file.endswith(".jpg"):
        file_name = os.path.join(image_dir, file)
        processing(file_name, file[:-4])
