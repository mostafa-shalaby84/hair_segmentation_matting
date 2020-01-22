# Automatic Hair Segmentation and Matting

### Contents
- [Face detection and normalization](#normalization)
- [Image Segmentation](#segmentation)
- [Image Matting](#matting)
- [Test](#Test)

## Normalization

The first step of the project is to detect a face region of the input image and normalize the images for future processing. We have adopted OpenCV DNN Facedetector. And based on the detection result the original image will be normalized as (512,512) size.

## Segmentation
The next step is to segment the image into several parts such as background, body, hat, and hair region. By used BiseNet architecture and Pytorch Framework we have built the segmentation model.
You can download [face-parsing pre-trained model](https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812) and save it in `models`, rename it as 'segmentation.pth'.

## Matting
Based on the segmented result, we have prepared the tripmap image for image matting. By used VGG16 architecture and Pytorch Framework we have built the matting model.
You can download [matting pre-trained model](https://github.com/huochaitiantang/pytorch-deep-image-matting/releases/download/v1.4/stage1_sad_54.4.pth) and save it in `models`, rename it as 'matting.pth'.

## Test
You have to prepare the images for testing, and copy them into the “images” folder. The result file will be saved in the “result” folder. 
```Shell
# evaluate using GPU
python test.py
```

| Original Image | Segmented and Alpha Image | Composition Image |
|---|---|---|
|![image](https://github.com/mostafa-shalaby84/hair_segmentation_matting/blob/master/result/00009_!real.png) |![image](https://github.com/mostafa-shalaby84/hair_segmentation_matting/blob/master/result/00009_alpha.png) |![image](https://github.com/mostafa-shalaby84/hair_segmentation_matting/blob/master/result/00009_composition.png)
|![image](https://github.com/mostafa-shalaby84/hair_segmentation_matting/blob/master/result/6_!real.png) |![image](https://github.com/mostafa-shalaby84/hair_segmentation_matting/blob/master/result/6_alpha.png) |![image](https://github.com/mostafa-shalaby84/hair_segmentation_matting/blob/master/result/6_composition.png)
|![image](https://github.com/mostafa-shalaby84/hair_segmentation_matting/blob/master/result/7_!real.png) |![image](https://github.com/mostafa-shalaby84/hair_segmentation_matting/blob/master/result/7_alpha.png) |![image](https://github.com/mostafa-shalaby84/hair_segmentation_matting/blob/master/result/7_composition.png)
