import cv2
from ultralytics import YOLO
import numpy as np
from threading import Thread
from queue import Queue
from lstm_model import lstmmodel
import os
def preprocess_image(image):
    # 自适应直方图均衡化
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    enhanced_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    # 拉普拉斯锐化
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    sharpened = cv2.filter2D(enhanced_image, -1, kernel)
    
    return sharpened
cnt_flag=1
kk=0
def capture_frames(url, frame_queue1,frame_queue2,frame_queue3,frame_queue4,frame_queue5):
    cap = cv2.VideoCapture(url)
    global cnt_flag
    global kk

    while True:
        kk+=1
        print("进入得到视频第"+str(kk))
        ret, frame = cap.read()

        #五轮循环对应连续五帧的图片，这样才可以输入lstm检测到（5*34）
        #要求每一个序列比另一个序列提前五帧，所以可以，并且因为queue1一定最先满，所以只需要用queue1来判断
        if not ret:
            break
        frame = cv2.resize(frame, (640, 384))
        if cnt_flag==1:
            if not frame_queue1.full():
                frame_queue1.put(frame)
            cnt_flag+=1

        elif cnt_flag==2:
            if not frame_queue1.full():
                frame_queue1.put(frame) 
                frame_queue2.put(frame)
            cnt_flag+=1

        elif cnt_flag==3:
            if not frame_queue1.full():
                frame_queue1.put(frame)   
                frame_queue2.put(frame)   
                frame_queue3.put(frame)      
            cnt_flag+=1

        elif cnt_flag==4:
            if not frame_queue1.full():
                frame_queue1.put(frame)            
                frame_queue2.put(frame)   
                frame_queue3.put(frame)      
                frame_queue4.put(frame)
            cnt_flag+=1

        else:
            if not frame_queue1.full():
                frame_queue1.put(frame)            
                frame_queue2.put(frame)   
                frame_queue3.put(frame)      
                frame_queue4.put(frame)
                frame_queue5.put(frame)
    cap.release()

cnt=0
def detect_fall(model, frame_queue1,frame_queue2,frame_queue3,frame_queue4,frame_queue5, output_queue):

    while True:
        global cnt
        # cnt+=1
        # print("进入detect第"+str(cnt))
        #因为queue5一定是最短的，所以只判断queue5
        if not frame_queue5.empty():
            frame1 = frame_queue1.get()
            frame2 = frame_queue2.get()
            frame3 = frame_queue3.get()
            frame4 = frame_queue4.get()
            frame5 = frame_queue5.get()
            framelist=[frame1,frame2,frame3,frame4,frame5]
            if frame1 is None or frame2 is None or frame3 is None or frame4 is None or frame5 is None:
                break
            kpts_arr=np.zeros((5,34))
            kpts_normalized=np.zeros(34)
            #false说明有人，不会出现没人，为空的情况
            loc_xy_flag=False
            for i in range(len(framelist)):
                preprocessed_frame=preprocess_image(framelist[i])
                det_params = {
                    'source': preprocessed_frame,
                    'imgsz': (640, 384),
                    'conf': 0.3,
                    'iou': 0.3
                }
                #这里假设只有一个人
                results=model(**det_params)
                results=results[0]
                kpts = results.keypoints[0].xy.numpy()
                # print("********************************")
                # print(kpts)
                # print(kpts.size)
                # print("********************************")
                #这里是检测人是不是存在的情况
                if kpts.size ==0:
                    if i <4:
                        kpts=np.zeros(34)
                    else:
                        loc_xy_flag=True
                        break
                
                kpts=kpts.flatten()
                # print("####################")
                # print(kpts)
                # print("####################")
                #对关键点进行归一化
                X_min = np.min(kpts)
                X_max = np.max(kpts)
                if X_max!=X_min:
                    # 应用最小-最大归一化公式
                    kpts_normalized = (kpts - X_min) / (X_max - X_min)

                kpts_arr[i,:]=kpts_normalized
                #下面这俩只取最后一帧的内容,能到这一步来说明最后一帧是可以检测到人的
                if i==4:
                    annotated_frame = results.plot()
                    loc_xy = results.boxes[0].xyxy[0].to(int).tolist()[:2]
            #如果没人，就进入下一个循环，不继续往下走了
            if loc_xy_flag:
                continue
            kpts_arr_3d = np.expand_dims(kpts_arr, axis=0)
            fall=lstmmodel.predict(kpts_arr_3d)
            threshold=0.5
            
            if fall < threshold:
                cv2.putText(annotated_frame, f'state: normal.', tuple(loc_xy), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 255, 0), 2)
            else:
                cv2.putText(annotated_frame, f'state: fall!!', tuple(loc_xy), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)



            
            # save_folder = 'img'

            # # 确保文件夹存在
            # if not os.path.exists(save_folder):
            #     os.makedirs(save_folder)
            # filename = f'image_{cnt:04d}.jpg'  # 使用02d来保证两位数的编号，例如01, 02, ...
            # # 完整的保存路径
            # save_path = os.path.join(save_folder, filename)
            # # 保存图片
            # cv2.imwrite(save_path, annotated_frame)
            # cnt+=1
            if not output_queue.full():
                output_queue.put(annotated_frame)
        else:
            print("所有都读取完毕")


def main():
    # model = YOLO(model='ultralytics/yolov8s_pose_last.pt')
    model = YOLO(model = "yolov8x-pose")
    
    url="D:\\BLTM\\reaction\\meeting_01.mp4"
    url=0
    url='13.mp4'
    frame_queue1 = Queue(maxsize=1000)
    frame_queue2 = Queue(maxsize=1000)
    frame_queue3 = Queue(maxsize=1000)
    frame_queue4 = Queue(maxsize=1000)
    frame_queue5 = Queue(maxsize=1000)
    output_queue = Queue(maxsize=10)
    

    capture_thread = Thread(target=capture_frames, args=(url, frame_queue1,frame_queue2,frame_queue3,frame_queue4,frame_queue5))
    detect_thread = Thread(target=detect_fall, args=(model, frame_queue1,frame_queue2,frame_queue3,frame_queue4,frame_queue5, output_queue))



    capture_thread.start()
    detect_thread.start()

    while True:
        if not output_queue.empty():
            annotated_frame = output_queue.get()
            # cv2.imshow("YOLOV8", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture_thread.join()
    frame_queue1.put(None)
    frame_queue2.put(None)
    frame_queue3.put(None)
    frame_queue4.put(None)
    frame_queue5.put(None)
    detect_thread.join()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
