
import os
import streamlit as st
# import ner_model as zwk
import pickle
import ollama
from transformers import BertTokenizer
import torch
import py2neo
import random
import re

from pathlib import Path
from openai import OpenAI
import time

client = OpenAI(
    api_key="sk-8cBOcDIxBALIOxfr4IfPzZdx5YOlVBgejfJ4TDpDNbmiAdDB",
    base_url="https://api.moonshot.cn/v1",
)

def generate_prompt(personal_info, medical_history, gait_analysis_result):
    gait_risk_factor = "步态异常检测结果为异常，跌倒风险增加。" if gait_analysis_result == "abnormal" else "步态异常检测结果为正常。"
    return f"""
    根据以下个人基本信息和病例信息，评估跌倒风险：

    **个人基本信息**
    {personal_info}

    **病例信息**
    {medical_history}
    
    **步态检测结果**
    {gait_risk_factor}

    请根据以下步骤评估跌倒风险：
    1. 分析个人基本信息，识别潜在风险因素：
       - 年龄和性别
       - 身高和体重
       - 视力状况
       - 平衡能力
       - 是否有骨折或骨质疏松史
       - 是否患有慢性疾病（如高血压、糖尿病、帕金森病等）
       - 足部健康状况（如足部畸形、溃疡等）
    2. 结合病例信息，确定相关的风险因子：
       - 是否有跌倒史
       - 当前的药物使用情况
       - 心理健康状况（如焦虑、抑郁）
       - 饮酒情况
    3. 结合环境因素，确定相关的风险因子：
       - 家中是否有适当的照明
       - 地面是否平整
    4. 结合步态异常检测结果，考虑步态异常对跌倒风险的影响。
    5. 量化每个风险因子的影响，并计算总体跌倒风险指数（0-100），并给出主要风险因素的解释。

    **输出格式（你每次的回答只需要按照此输出格式进行输出，不要添加其他的信息，只需要给出一个风险指数）**
    ```
    跌倒风险指数: [风险指数]
    主要风险因素:
    - [风险因素1]
    - [风险因素2]
    - ...
    ```
    ```
    """

def extract_fall_risk_index(response_text):
    match = re.search(r"跌倒风险指数:\s*(\d+)", response_text)
    if match:
        risk_index = int(match.group(1))
        return risk_index
    else:
        raise ValueError("未找到跌倒风险指数")

def Fall_Risk_Evaluation(personal_info, medical_history, gait_analysis_result):
    prompt = generate_prompt(personal_info, medical_history, gait_analysis_result)
    response = ollama.generate(model='llama3', prompt=prompt, options={"temperature":0.3})['response']
    print(f'跌倒风险评估结果:{response}')
    return response

def Fall_Risk_Evaluation_by_kimi_api(client, personal_info, medical_history, gait_analysis_result):
    prompt = generate_prompt(personal_info, medical_history, gait_analysis_result)
    response = client.chat.completions.create(
        model='moonshot-v1-8k',
        messages=[
            {"role": "system",
             "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话,你擅长根据用户的个人信息以及病历信息预测出稳定、科学的跌倒风险指数。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    print(f'跌倒风险评估结果:{response.choices[0].message.content}')
    return response.choices[0].message.content

# 数据
test_cases = [
    {
        "personal_info": "- 姓名: 李明\n- 年龄: 75岁\n- 性别: 男\n- 身高: 165cm\n- 体重: 65kg\n- 其他信息: 使用拐杖",
        "medical_info": "- 既往病史: 高血压, 糖尿病\n- 近期病情: 无\n- 用药情况: 使用降压药, 控糖药",
        "gait_analysis_result": "normal"  # 正常步态检测
    },
    {
        "personal_info": "- 姓名: 李明\n- 年龄: 75岁\n- 性别: 男\n- 身高: 165cm\n- 体重: 65kg\n- 其他信息: 使用拐杖",
        "medical_info": "- 既往病史: 高血压, 糖尿病\n- 近期病情: 无\n- 用药情况: 使用降压药, 控糖药",
        "gait_analysis_result": "abnormal"  # 异常步态检测
    },
    {
        "personal_info": "- 姓名: 张华\n- 年龄: 80岁\n- 性别: 女\n- 身高: 160cm\n- 体重: 55kg\n- 其他信息: 有轻度视力障碍",
        "medical_info": "- 既往病史: 骨质疏松, 关节炎\n- 近期病情: 近期摔倒过一次\n- 用药情况: 钙片, 维生素D, 抗炎药",
        "gait_analysis_result": "normal"  # 正常步态检测
    },
    {
        "personal_info": "- 姓名: 张华\n- 年龄: 80岁\n- 性别: 女\n- 身高: 160cm\n- 体重: 55kg\n- 其他信息: 有轻度视力障碍",
        "medical_info": "- 既往病史: 骨质疏松, 关节炎\n- 近期病情: 近期摔倒过一次\n- 用药情况: 钙片, 维生素D, 抗炎药",
        "gait_analysis_result": "abnormal"  # 异常步态检测
    },
    {
        "personal_info": "- 姓名: 王芳\n- 年龄: 68岁\n- 性别: 女\n- 身高: 158cm\n- 体重: 60kg\n- 其他信息: 无",
        "medical_info": "- 既往病史: 高血压\n- 近期病情: 近期头晕\n- 用药情况: 降压药",
        "gait_analysis_result": "normal"  # 正常步态检测
    },
    {
        "personal_info": "- 姓名: 王芳\n- 年龄: 68岁\n- 性别: 女\n- 身高: 158cm\n- 体重: 60kg\n- 其他信息: 无",
        "medical_info": "- 既往病史: 高血压\n- 近期病情: 近期头晕\n- 用药情况: 降压药",
        "gait_analysis_result": "abnormal"  # 异常步态检测
    },
    {
        "personal_info": "- 姓名: 赵强\n- 年龄: 85岁\n- 性别: 男\n- 身高: 170cm\n- 体重: 70kg\n- 其他信息: 独居",
        "medical_info": "- 既往病史: 帕金森病\n- 近期病情: 步态不稳\n- 用药情况: 帕金森药物",
        "gait_analysis_result": "normal"  # 正常步态检测
    },
    {
        "personal_info": "- 姓名: 赵强\n- 年龄: 85岁\n- 性别: 男\n- 身高: 170cm\n- 体重: 70kg\n- 其他信息: 独居",
        "medical_info": "- 既往病史: 帕金森病\n- 近期病情: 步态不稳\n- 用药情况: 帕金森药物",
        "gait_analysis_result": "abnormal"  # 异常步态检测
    },
    {
        "personal_info": "- 姓名: 陈丽\n- 年龄: 73岁\n- 性别: 女\n- 身高: 155cm\n- 体重: 52kg\n- 其他信息: 曾骨折",
        "medical_info": "- 既往病史: 骨质疏松, 糖尿病\n- 近期病情: 无\n- 用药情况: 钙片, 维生素D, 控糖药",
        "gait_analysis_result": "normal"  # 正常步态检测
    },
    {
        "personal_info": "- 姓名: 陈丽\n- 年龄: 73岁\n- 性别: 女\n- 身高: 155cm\n- 体重: 52kg\n- 其他信息: 曾骨折",
        "medical_info": "- 既往病史: 骨质疏松, 糖尿病\n- 近期病情: 无\n- 用药情况: 钙片, 维生素D, 控糖药",
        "gait_analysis_result": "abnormal"  # 异常步态检测
    }
]

# 主函数
def main():
    for i, case in enumerate(test_cases, 1):
        print(f"用例 {i}:")
        personal_info = case["personal_info"]
        medical_info = case["medical_info"]
        gait_analysis_result = case["gait_analysis_result"]
        print(personal_info,medical_info,gait_analysis_result)
        # result = Fall_Risk_Evaluation_by_kimi_api(client, personal_info, medical_info, gait_analysis_result) # kimi_api
        result = Fall_Risk_Evaluation(personal_info, medical_info, gait_analysis_result)   # ollma
        # 提取风险指数
        try:
            risk_index = extract_fall_risk_index(result)
            print(f"提取的跌倒风险指数为: {risk_index}")
        except ValueError as e:
            print(e)
            print(result)

if __name__ == "__main__":
    main()