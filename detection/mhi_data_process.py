#先读取所有视频（跌倒，不跌倒），并且把视频每一帧处理到的
import cv2
import os
import numpy as np
from mhi_model import MHI


def frame_resize(frame):
    frame=cv2.resize(frame,(400,300))
    return frame




def save_mhi(save_path,video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    ret,frame=cap.read()
    if not ret:
        return 
    processed_frame=frame_resize(frame)
    mhi=MHI(processed_frame,tau=200,xi=20,delta=10,t=10)
    cnt=0
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        processed_frame=frame_resize(frame)
        mhi_frame=mhi.getimag(processed_frame)
        if frame is None:
            print("Error: Image not loaded.")
            break
        else:
            cv2.imwrite(save_path+str(cnt)+'.jpg',mhi_frame)
        cnt+=1
    cap.release()




def save_fall_mhi(fall_dir):
    cnt=0
    for filename in os.listdir(fall_dir):
        fall_fp= os.path.join(fall_dir, filename)
        if fall_fp.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            save_path='img/FALL/'+str(cnt)+'_'
            save_mhi(save_path,fall_fp)
        cnt+=1

def save_adl_mhi(adl_dir):
    cnt=0
    for filename in os.listdir(adl_dir):
        adl_fp= os.path.join(adl_dir, filename)
        if adl_fp.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            save_path='img/ADL/'+str(cnt)+'_'
            save_mhi(save_path,adl_fp)
        cnt+=1

def main():
    img_adl_dir = str(os.getcwd())+'/img/ADL'
    img_fall_dir = str(os.getcwd())+'/img/FALL'
    #创建或者删除旧文件
    if not os.path.exists(img_adl_dir):
        os.makedirs(img_adl_dir)
        print(f"目录 {img_adl_dir} 已被创建。")
    for filename in os.listdir(img_adl_dir):
        file_path = os.path.join(img_adl_dir, filename)
        # 检查是否是文件
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"文件 {filename} 已被删除。")

    if not os.path.exists(img_fall_dir):
        os.makedirs(img_fall_dir)
        print(f"目录 {img_fall_dir} 已被创建。")
    for filename in os.listdir(img_fall_dir):
        file_path = os.path.join(img_fall_dir, filename)
        # 检查是否是文件
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"文件 {filename} 已被删除。")
    

    adl_dir = str(os.getcwd())+'/video/ADL'
    fall_dir = str(os.getcwd())+'/video/FALL'

    save_fall_mhi(fall_dir)
    save_adl_mhi(adl_dir)

    



if __name__=="__main__":
    main()