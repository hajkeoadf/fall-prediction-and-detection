#先读取所有视频（跌倒，不跌倒），并且把视频每一帧处理到的
import cv2
import os
import numpy as np
from ultralytics import YOLO
import pickle as pkl

# 然后把每一帧的关键点，然后把每一帧的关键点都读取出来，以一秒为准，一般采用25帧或者30帧

def frame_process(frame):
    frame=cv2.resize(frame,(400,300))
    img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    enhanced_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    # 拉普拉斯锐化
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    sharpened = cv2.filter2D(enhanced_image, -1, kernel)
    return sharpened


def get_kpts(frame):
    model = YOLO(model = "yolov8x-pose")
    det_params = {
        'source': frame,
        'conf': 0.3,
        'iou': 0.3
    }
    #这里假设只有一个人
    results=model(**det_params)
    results=results[0]
    kpts = results.keypoints[0].xy.numpy()
    
    if kpts.size==0:
        kpts=np.zeros((17,2))
    else:
        kpts=np.squeeze(kpts)
    print("kpts的大小为：",kpts.shape)
    return kpts

def process_fall(fall_dir):
    fall_kpts_list=[]
    for filename in os.listdir(fall_dir):
        fall_fp= os.path.join(fall_dir, filename)
        if fall_fp.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            cap = cv2.VideoCapture(fall_fp)
            if not cap.isOpened():
                print(f"Error: Could not open video {fall_fp}")
                continue

            #存储每指定帧数的关键点坐标信息
            fps=30
            kpts_num=17
            layer=2
            kpts_fake_img=np.zeros((fps, kpts_num,layer))

            #开始处理帧
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                #对每帧图片进行图片处理
                processed_frame=frame_process(frame)

                #得到每帧图片的关键点信息
                kpts=get_kpts(processed_frame)

                frame_cnt=frame_count%fps
                kpts_fake_img[frame_cnt,:,:]=kpts

                if frame_cnt==fps-1:#到了指定帧数就存一次
                    fall_kpts_list.append(kpts_fake_img)
                frame_count += 1

            cap.release()
            print(f"{fall_fp} videos have been finished.")
    print("Finished processing all fall videos.")
    #list转换为numpy的数据
    fall_kpts_array = np.stack(fall_kpts_list, axis=0)
    print("跌倒行为的数据大小：",fall_kpts_array.shape)
    return fall_kpts_array



def process_adl(adl_dir):
    adl_kpts_list=[]
    for filename in os.listdir(adl_dir):
        adl_fp= os.path.join(adl_dir, filename)
        if adl_fp.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            cap = cv2.VideoCapture(adl_fp)
            if not cap.isOpened():
                print(f"Error: Could not open video {adl_fp}")
                continue
             #存储每指定帧数的关键点坐标信息
            fps=30
            kpts_num=17
            layer=2
            kpts_fake_img=np.zeros((fps, kpts_num,layer))
            #开始处理帧
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                #对每帧图片进行图片处理
                processed_frame=frame_process(frame)

                #得到每帧图片的关键点信息
                kpts=get_kpts(processed_frame)

                frame_cnt=frame_count%fps
                kpts_fake_img[frame_cnt,:,:]=kpts
                if frame_cnt==fps-1:#到了指定帧数就存一次
                    adl_kpts_list.append(kpts_fake_img)
                frame_count += 1

            cap.release()
            print(f"{adl_fp} videos have been finished.")

    print("Finished processing all adl videos.")
    #list转换为numpy的数据
    adl_kpts_array = np.stack(adl_kpts_list, axis=0)
    print("正常行为的数据大小：",adl_kpts_array.shape)
    return adl_kpts_array

def main():
    adl_dir = str(os.getcwd())+'/video/ADL'
    fall_dir = str(os.getcwd())+'/video/FALL'
    adl_kpts_array=process_adl(adl_dir)
    fall_kpts_array=process_fall(fall_dir)
    #将xy轴坐标设置为两通道
    adl_kpts_array = adl_kpts_array.transpose((0, 3, 1, 2))
    fall_kpts_array = fall_kpts_array.transpose((0, 3, 1, 2))


    #将处理得到的数据存储入label文件夹
    label_dir = str(os.getcwd())+'/label'
    print(label_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
        print(f"目录 {label_dir} 已被创建。")
    for filename in os.listdir(label_dir):
        file_path = os.path.join(label_dir, filename)
        # 检查是否是文件
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"文件 {filename} 已被删除。")

    file_name='fall.pkl'
    file_path = os.path.join(label_dir, file_name)
    with open(file_path, 'wb') as f:
        pkl.dump(fall_kpts_array,f)
        print(f"文件 {file_name} 已被创建。")

    file_name='adl.pkl'
    file_path = os.path.join(label_dir, file_name)
    with open(file_path, 'wb') as f:
        pkl.dump(adl_kpts_array,f)
        print(f"文件 {file_name} 已被创建。")


if __name__=="__main__":
    main()