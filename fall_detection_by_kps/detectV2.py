import cv2
from ultralytics import YOLO
import numpy as np
# import torch
from threading import Thread
from queue import Queue




def preprocess_image(image):
    # # 直方图均衡化
    # img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # enhanced_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    # 自适应直方图均衡化
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    enhanced_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    # 拉普拉斯锐化
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    sharpened = cv2.filter2D(enhanced_image, -1, kernel)
    
    return sharpened

def capture_frames(url, frame_queue):
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 384))

        if not frame_queue.full():
            frame_queue.put(frame)
    cap.release()

def detect_fall(model, frame_queue, output_queue):
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:
                break

            preprocessed_frame = preprocess_image(frame)
            det_params = {
                'source': preprocessed_frame,
                'imgsz': (640, 384),
                'conf': 0.3,
                'iou': 0.3
            }

            results = model(**det_params)
            annotated_frame = results[0].plot()

            fall = [0] * results[0].boxes.shape[0]
            #识别到的人数
            for i in range(results[0].boxes.shape[0]):
                print("%%%%%%%%%%%%%%%%%%%")
                print(i)
                print("%%%%%%%%%%%%%%%%%%%")
                kpts = results[0].keypoints[i].xy
                # print(type(kpts))




                #接下来判别这一块儿全改
                if (kpts[0][5][1] + kpts[0][6][1]) / 2 >= (kpts[0][11][1] + kpts[0][12][1]) / 2:
                    if kpts[0][5][1] > 0 and kpts[0][6][1] > 0 and kpts[0][11][1] > 0 and kpts[0][12][1] > 0:
                        fall[i] = 1
                else:
                    dy = (kpts[0][11][1] + kpts[0][12][1]) / 2 - (kpts[0][5][1] + kpts[0][6][1]) / 2
                    dx = abs((kpts[0][11][0] + kpts[0][12][0]) / 2 - (kpts[0][5][0] + kpts[0][6][0]) / 2)
                    if dx > 0 and dy > 0:
                        deg = np.arctan((dy / dx).item()) * 180 / np.pi
                        if deg < 50:
                            fall[i] = 1

                loc_xy = results[0].boxes[i].xyxy[0].to(int).tolist()[:2]
                loc_xy[1] -= 30
                if fall[i] == 0:
                    cv2.putText(annotated_frame, f'state: normal.', tuple(loc_xy), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 255, 0), 2)
                else:
                    cv2.putText(annotated_frame, f'state: fall!!', tuple(loc_xy), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            if not output_queue.full():
                output_queue.put(annotated_frame)

def main():
    # model = YOLO(model='ultralytics/yolov8s_pose_last.pt')
    model = YOLO(model = "yolov8x-pose")
    
    url="D:\\BLTM\\reaction\\meeting_01.mp4"
    url=0
    frame_queue = Queue(maxsize=10)
    output_queue = Queue(maxsize=10)

    capture_thread = Thread(target=capture_frames, args=(url, frame_queue))
    detect_thread = Thread(target=detect_fall, args=(model, frame_queue, output_queue))

    capture_thread.start()
    detect_thread.start()

    while True:
        if not output_queue.empty():
            annotated_frame = output_queue.get()
            cv2.imshow("YOLOV8", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture_thread.join()
    frame_queue.put(None)
    detect_thread.join()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
