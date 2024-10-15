from skeletonDetect import *
import numpy as np
from joblib import load
# 环境叫tensor


def extract_alternating_sequence(f2, f3):
    valid_sequence = []

    # 以 f2 开始，f3 结束，必须交替出现
    f2_idx = 0
    f3_idx = 0

    # 遍历 f2 和 f3 列表，确保以 f2 开始，并找到最近的 f3
    while f2_idx < len(f2) and f3_idx < len(f3):
        # 确保 f2 在 f3 之前
        if f2[f2_idx] < f3[f3_idx]:
            # 寻找下一个 f2，直到 f3 大于当前 f2
            while f2_idx < len(f2) - 1 and f2[f2_idx + 1] < f3[f3_idx]:
                f2_idx += 1  # 取最后一个 f2
            valid_sequence.append(f2[f2_idx])

            # 寻找下一个 f3，直到它大于当前 f2
            while f3_idx < len(f3) - 1 and f3[f3_idx + 1] <= f2[f2_idx]:
                f3_idx += 1  # 取最后一个 f3
            valid_sequence.append(f3[f3_idx])

            # 移动到下一个 f2 和 f3
            f2_idx += 1
            f3_idx += 1
        else:
            # 如果 f3 在 f2 之前，跳过 f3
            f3_idx += 1

    # 返回最终的交替序列
    return valid_sequence


def median_average_filter(signal, window_size):
    # 确保窗口尺寸是奇数
    assert window_size % 2 == 1, "Window size must be odd."
    
    # 信号的长度
    signal_length = len(signal)
    
    # 初始化过滤后的信号
    filtered_signal = np.zeros(signal_length)
    
    # 半窗口大小
    half_window = window_size // 2
    
    # 遍历信号
    for i in range(half_window, signal_length - half_window):
        # 获取窗口内的值
        window = signal[i - half_window : i + half_window + 1]
        
        # 删除窗口内的最大值和最小值
        window = np.delete(window, np.argmax(window))
        window = np.delete(window, np.argmin(window))
        
        # 计算剩余值的平均
        filtered_signal[i] = np.mean(window)
    
    # 处理边界条件，这里简单地使用最近的非边界值
    filtered_signal[:half_window] = filtered_signal[half_window]
    filtered_signal[-half_window:] = filtered_signal[-half_window-1]
    
    return filtered_signal


def smooth_keypoints(keypoints, window_size):
    # 将 keypoints 转换为 NumPy 数组
    keypoints = np.array(keypoints)

    # 确保窗口尺寸是奇数
    assert window_size % 2 == 1, "Window size must be odd."
    
    # 初始化过滤后的关键点
    smoothed_keypoints = np.zeros_like(keypoints)
    
    frame_count, keypoint_count, _ = keypoints.shape
    for keypoint_index in range(keypoint_count):
        for coordinate_index in [0, 1]:  # 只对 x 和 y 坐标进行平滑
            # 提取单个关键点的 x 或 y 坐标序列
            coordinate_series = keypoints[:, keypoint_index, coordinate_index]
            # 对坐标序列应用平滑函数
            smoothed_coordinate_series = median_average_filter(coordinate_series, window_size)
            # 将平滑后的坐标序列存回原位置
            smoothed_keypoints[:, keypoint_index, coordinate_index] = smoothed_coordinate_series
        # 置信度保持不变
        smoothed_keypoints[:, keypoint_index, 2] = keypoints[:, keypoint_index, 2]
            
    return smoothed_keypoints



def calculatethre(signal, lambda_up, lambda_down):
    ignore_percentage = 0.1  # 忽略头部和尾部的百分比
    # 忽略头部和尾部一定比例的数据点
    ignore_points = int(len(signal) * ignore_percentage)
    trimmed_signal = signal[ignore_points:-ignore_points]

    # 计算信号的最大值和最小值
    max_value = max(trimmed_signal)
    min_value = min(trimmed_signal)

    # 计算阈值
    thre_up = min_value + (max_value - min_value) * lambda_up
    thre_down = min_value + (max_value - min_value) * lambda_down

    return thre_up, thre_down


# , signal_15, lambdaUp15, lambdaDown15
def subtaskSegmentationShoulder(signal_x, lambdaUpX, lambdaDownX, signal_y, lambdaUpY, lambdaDownY):
    threUpX, threDownX = calculatethre(signal_x, lambdaUpX, lambdaDownX)
    currentState = 0
    resultX = []
    startingPointX = int(0.3 * len(signal_x))
    endingPointX = int(1 * len(signal_x))
    for i in range(startingPointX, endingPointX-1):
        if signal_x[i] < threUpX and signal_x[i+1] >= threUpX:
            if currentState == 0:
                resultX.append(i+1)
                currentState = 1
            elif currentState == 1:
                resultX[-1] = i+1
        if signal_x[i] > threDownX and signal_x[i+1] <= threDownX and len(resultX) != 0:
            if currentState == 1:
                resultX.append(i+1)
                currentState = 0
            elif currentState == 0:
                resultX[-1] = i+1

    if len(resultX) % 2 != 0 and len(resultX) > 3:
        resultX.pop()

    filtered_resultX = []
    for i in range(0, len(resultX), 2):
        if (i+1) >= len(resultX):
            filtered_resultX.append(resultX[i])
            break
        diff = resultX[i + 1] - resultX[i]
        # 检查差值是否在 80 到 150 之间
        if 20 <= diff <= 150:
            # 如果满足条件，则保留这对元素
            filtered_resultX.append(resultX[i])
            filtered_resultX.append(resultX[i + 1])

    threUpY, threDownY = calculatethre(signal_y, lambdaUpY, lambdaDownY)
    currentState = 0
    resultY = []
    front15 = int(0.3*len(signal_y))
    back15 = int(0.6*len(signal_y))
    for i in range(len(signal_y)-1):
        if i < front15 - 1 and currentState == 0:
            # 检查前15%的数据中小于threUpY到大于等于threUpY的点
            if signal_y[i] > threDownY and signal_y[i + 1] <= threDownY:
                resultY.append(i + 1)
                currentState = 1
        elif i >= back15 and currentState == 1:
            # 检查后15%的数据中大于threDownY到小于等于threDownY的点
            if signal_y[i] < threUpY and signal_y[i + 1] >= threUpY:
                resultY.append(i + 1)
                currentState = 0
    return filtered_resultX, threUpX, threDownX, resultY, threUpY, threDownY


def subtaskSegmentationAnkle(signal, lambdaUp, lambdaDown, f1, f2, f3, f4):
    resultWalk = []
    threWalkUp, threWalkDown = calculatethre(signal, lambdaUp, lambdaDown)
    currentState = 0
    step1 = 0
    for i in range(f1, f2-1):
        if signal[i] < threWalkUp and signal[i+1] >= threWalkUp:
            if currentState == 0:
                resultWalk.append(i+1)
                currentState = 1
                step1 += 1
        if signal[i] > threWalkDown and signal[i+1] <= threWalkDown and len(resultY)!= 0:
            if currentState == 1:
                resultWalk.append(i+1)
                currentState = 0
                step1 += 1

    currentState = 0
    step2 = 0
    resultBack = []
    for i in range(f3, f4-1):
        if signal[i] < threWalkUp and signal[i+1] >= threWalkUp:
            if currentState == 0:
                resultBack.append(i+1)
                currentState = 1
                step2 += 1
        if signal[i] > threWalkDown and signal[i+1] <= threWalkDown and len(resultY)!= 0:
            if currentState == 1:
                resultBack.append(i+1)
                currentState = 0
                step2 += 1
    return resultWalk, step1, resultBack, step2


def caculateParam(f0, f1, f2, f3, f4, s, fr, step1, step2):
    standUpTime = (f1 - f0) / fr
    turningTime = (f3 - f2) / fr
    velocity = s * fr / ((f2 - f1)+(f4 - f3))
    stepLength = s / (step1+step2)
    cadence = (step1+step2) * fr / ((f2-f1)+(f4-f3))

    return standUpTime, turningTime, velocity, stepLength, cadence

if __name__ == '__main__':
    path = "../dataset/videos/ori_videos/006_color.mp4"
    model_file_path = 'svm_model_rbf.joblib'
    svm_model = load(model_file_path)

    keypoints = get_skeleton(path)
    keypoints = smooth_keypoints(keypoints, window_size=5)

    shoulder_x_diff = []
    shoulder_y_avg = []
    ankele_x_diff = []
    for keypoint in keypoints:
        shoulder_x_diff.append((keypoint[5][1]-keypoint[6][1])*640)
        shoulder_y_avg.append((keypoint[5][0]+keypoint[6][0])*480/2)
        ankele_x_diff.append((keypoint[15][1]-keypoint[16][1])*640)
    # # 在视频上可视化骨架提取结果
    # cap = cv2.VideoCapture(path)
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     # 读取frame的大小
    #     if not ret:
    #         break
    #     for keypoint in keypoints:
    #         #需要乘上frame的大小
    #         yL, xL, _ = keypoint[5]
    #         cv2.circle(frame, (int(xL*frame.shape[1]), int(yL*frame.shape[0])), 5, (255, 255, 0), -1)
    #         yR, xR, _ = keypoint[6]
    #         cv2.circle(frame, (int(xR*frame.shape[1]), int(yR*frame.shape[0])), 5, (0, 255, 0), -1)
    #     cv2.imshow('Skeleton', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    resultX, threUpX, threDownX, resultY, threUpY, threDownY = subtaskSegmentationShoulder(shoulder_x_diff, 0.5, 2/3, shoulder_y_avg, 0.3, 1/3) 
    f0 = 0
    f1 = resultY[0]
    f2 = resultX[0]
    f3 = resultX[1]
    f4 = resultX[2]
    f5 = resultY[1]
    resultWalk, step1, resultBack, step2 = subtaskSegmentationAnkle(ankele_x_diff, 0.5, 0.4, f1, f2, f3, f4) 

    s = 6
    fr = 24
    standUpTime, turningTime, velocity, stepLength, cadence = caculateParam(f0, f1, f2, f3, f4, s, fr, step1, step2)
    print(f"step1: {step1}")
    print(f"step2: {step2}")
    print(f"stepLength: {stepLength}")
    print(f"velocity: {velocity}")
    print(f"cadence: {cadence}")
    print(f"standUpTime: {standUpTime}")
    print(f"turningTime: {turningTime}")


    features = [step1, step2, stepLength, velocity, cadence, standUpTime, turningTime]
    features = np.array(features).reshape(1, -1)
    prediction = svm_model.predict(features)
    print(f'Prediction: {prediction}')

    # frames = range(0, len(shoulder_x_diff))
    # plt.figure(figsize=(10, 5))
    # plt.plot(frames, shoulder_y_avg, label='Average Shoulder Y')
    # plt.plot(frames, shoulder_x_diff, label='Difference in Shoulder X')
    # plt.plot(frames, ankele_x_diff, label='Difference in Ankle X')
    # plt.scatter(resultX, [shoulder_x_diff[i] for i in resultX], color='green', marker='o', s=50, label='resultX')
    # plt.scatter(resultY, [shoulder_y_avg[i] for i in resultY], color='green', marker='o', s=50, label='resultY')
    # plt.scatter(resultWalk, [ankele_x_diff[i] for i in resultWalk], color='blue', marker='o', s=50, label='resultWalk')
    # plt.scatter(resultBack, [ankele_x_diff[i] for i in resultBack], color='yellow', marker='o', s=50, label='resultBack')

    # plt.axhline(y=threUpX, color='purple', linestyle='--', linewidth=1, label='thre Up X')
    # plt.axhline(y=threDownX, color='orange', linestyle='--', linewidth=1, label='thre Down X')

    # plt.axhline(y=threUpY, color='purple', linestyle='--', linewidth=1, label='thre Up Y')
    # plt.axhline(y=threDownY, color='orange', linestyle='--', linewidth=1, label='thre Down Y')

    # plt.xlabel('Frame')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.grid(True)
    # plt.show()







