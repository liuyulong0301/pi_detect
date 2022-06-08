import numpy as np
import cv2
import random
import math
import time
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

# 左相机内参
left_camera_matrix = np.array([[616.3319, -0.9517, 288.5405],[0, 618.5550, 200.7788],[0,0,1]])
left_distortion = np.array([[0.1947, -0.0584,  0.0032, -7.8474e-04,  -0.4253]])

right_camera_matrix = np.array([[615.2792, -0.8063, 316.5238],[0, 618.1086, 207.2990],[0,0,1]])
right_distortion = np.array([[0.2250, -0.2018, 0.0039, -0.0031, -0.7356]])

R=np.array([[0.9998, -0.0139, 0.0152],[-0.0138, 0.9999, 0.0022],[-0.0153, -0.0020, 0.9999]])#旋转矩阵
T = np.array([-52.1912, 0.3723, 0.6168])#平移矩阵

size = (640, 480)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)

left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

def run(frame1: np.ndarray, model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
    '''
    # Variables to calculate FPS
    #counter, fps = 0, 0
    #start_time = time.time()
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10
    '''
    # Initialize the object detection model
    base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
    detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

    
    image = frame1
    #counter += 1
    #image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)
   
    return detection_result

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 480)  # 打开并设置摄像头

#WIN_NAME = 'Deep disp'
#cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

while True:
    
    start_time = time.time()
    
    ret, frame = cap.read()
    frame1 = frame[:, 0:640, :]
    frame2 = frame[:, 640:1280, :]

    imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # 将BGR格式转换成灰度图片
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

    imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
    imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)
    #cv2.imshow("imageL",imageL)
    # BM
    numberOfDisparities = ((640 // 8) + 15) & -16  # 640对应是分辨率的宽

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=9)  # 立体匹配
    stereo.setROI1(validPixROI1)
    stereo.setROI2(validPixROI2)
    stereo.setPreFilterCap(1)
    #stereo.setPreFilterCap(31)
    stereo.setBlockSize(31)
    stereo.setMinDisparity(0)
    stereo.setNumDisparities(numberOfDisparities)
    stereo.setTextureThreshold(10)
    stereo.setUniquenessRatio(15)
    stereo.setSpeckleWindowSize(1)
    #stereo.setSpeckleWindowSize(100)
    stereo.setSpeckleRange(1)
    #stereo.setSpeckleRange(32)
    stereo.setDisp12MaxDiff(1)


    disparity = stereo.compute(img1_rectified, img2_rectified)
    
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # 归一化函数算法

    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)  # 计算三维坐标数据值
    threeD = threeD * 16
    
    detection_result = run(frame1, 'efficientdet_lite0.tflite', 0, 640, 480, 4, False)
    #print(detection_result)
    
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        x = int(bbox.origin_x + bbox.width/2)
        y = int(bbox.origin_y + bbox.height/2)
        category = detection.classes[0]
        class_name = category.class_name
        
        print('\n像素坐标 x = %d, y = %d' % (x, y))
       
        print("世界坐标 xyz 是：", threeD[y][x][0] / 1000.0, threeD[y][x][1] / 1000.0, threeD[y][x][2] / 1000.0, "m")

        distance = math.sqrt(threeD[y][x][0] ** 2 + threeD[y][x][1] ** 2 + threeD[y][x][2] ** 2)
        distance = distance / 1000.0
        if x<240:
            print("左前方距离%s %f m"%(class_name,distance))
        elif x>=240 and x<=400:
            print("正前方距离%s %f m"%(class_name,distance))
        else:
            print("右前方距离%s %f m"%(class_name,distance))
        

    cv2.imshow("disp", disp)  # 显示深度图的双目画面
    
    end_time = time.time()
    use_time = end_time-start_time
    fps = 1/use_time
    print("use_time:%fs"%use_time)
    print("fps:",fps)
    
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyALLWindows()