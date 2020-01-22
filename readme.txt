Automatic Hair Segmentation and Matting

1. Face detection and Normalization
The first step of the project is to detect a face region of the input image and normalize the images for future processing. We have adopted OpenCV DNN Facedetector(Fig.1 a). And based on the detection result the original image will be normalized as (512,512) size (Fig.1 b).

2. Image Segmentation
The next step is to segment the image into several parts such as background, body, hat, and hair region (Fig.1 c). By used BiseNet architecture and Pytorch Framework we have built the segmentation model (face parsing model). Fig. 2.

3. Image Matting
Based on the segmented result, we have prepared the tripmap image(Fing.1. d) for image matting. By used VGG16 architecture and Pytorch Framework we have built the matting model.

4. Test
You have to prepare the images for testing, and copy them into the “images” folder. The result file will be saved in the “result” folder. 
>python test.py