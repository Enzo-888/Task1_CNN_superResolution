# -*- coding: utf-8 -*-
"""
从大型Numpy数据集中创建小型Demo数据集的工具脚本
=====================================================

功能:
- 读取指定路径的大型 .npy 文件。
- 从第一个维度（样本维度）中截取指定数量的样本。
- 将截取后的小型数据集保存到指定的输出路径。
- 对于2D的元数据（如mask, lat, lon），则直接进行复制。

如何使用:
1. 修改下面的 "USER CONFIGURATION" 部分。
2. 填入你的原始大文件的【完整路径】。
3. 填入你希望保存demo文件的【完整路径】。
4. 设置你想要的样本数量 `DEMO_SAMPLE_COUNT`。
5. 运行此脚本。
"""

import numpy as np
import os
import shutil

def create_demo_subset(input_path, output_path, num_samples):
    """
    读取一个.npy文件，截取前num_samples个样本，并保存到新文件。
    如果输入文件是2D的，则直接复制。

    Args:
        input_path (str): 输入的.npy文件的完整路径。
        output_path (str): 输出的demo .npy文件的完整路径。
        num_samples (int): 要截取的样本数量。
    """
    print(f"--- 正在处理: {os.path.basename(input_path)} ---")

    # --- 安全检查: 检查输入文件是否存在 ---
    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在，已跳过: {input_path}")
        return

    # --- 确保输出目录存在 ---
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    # --- 加载数据 ---
    print(f"正在从 '{input_path}' 加载数据...")
    original_data = np.load(input_path)
    print(f"原始数据形状: {original_data.shape}")

    # --- 判断数据维度，并进行处理 ---
    if original_data.ndim == 3:  # 例如 (1060, 400, 400)，这是需要截取的数据
        # 检查请求的样本数是否超过总数
        total_samples = original_data.shape[0]
        if num_samples > total_samples:
            print(f"警告: 请求的样本数 ({num_samples}) 超过了文件总样本数 ({total_samples})。")
            print(f"将使用所有可用样本。")
            num_samples = total_samples
        
        # --- 进行切片操作 ---
        demo_data = original_data[:num_samples, :, :]
        
        print(f"已截取前 {num_samples} 个样本。")
        print(f"新的Demo数据形状: {demo_data.shape}")
        
        # --- 保存新的Demo数据 ---
        np.save(output_path, demo_data)
        print(f"成功保存Demo文件至: {output_path}")

    elif original_data.ndim == 2:  # 例如 (400, 400)，这是mask, lat, lon等元数据
        print("检测到2D元数据，将直接复制文件。")
        shutil.copyfile(input_path, output_path)
        print(f"成功复制元数据至: {output_path}")
        
    else:
        print(f"警告: 不支持的数据维度 ({original_data.ndim})，已跳过此文件。")

    print("-" * (len(os.path.basename(input_path)) + 14))
    print()


if __name__ == "__main__":
    
    # ===================================================================
    #                           USER CONFIGURATION
    # ===================================================================
    # 1. 设置你想要的Demo数据集包含的样本数量
    DEMO_SAMPLE_COUNT = 150 

    # 2. 定义文件处理列表
    #    为每个要处理的文件，提供'input'和'output'的完整路径。
    #    !!! 请将下面的路径修改为您自己的实际路径 !!!
    file_processing_list = [
        {
            "description": "高分辨率训练数据",
            "input": "/data_new/OceanSR/split_ocean_dataset_seq/train/hr/ubar_hr_train.npy",
            "output": "/home/yc/Task1/data/u_bar/ubar_hr_train.npy"
        },
        {
            "description": "低分辨率训练数据",
            "input": "/data_new/OceanSR/split_ocean_dataset_seq/train/lr/ubar_sr_input_train.npy",
            "output": "/home/yc/Task1/data/u_bar/ubar_sr_input_train.npy"
        },
        # {
        #     "description": "元数据 - Mask",
        #     "input": "/data_new/OceanSR/split_ocean_dataset_seq/train/static/mask_hr_full.npy",
        #     "output": "/home/yc/Task1/data/v_bar/mask.npy"
        # },
        # {
        #     "description": "元数据 - Latitude",
        #     "input": "/data_new/OceanSR/split_ocean_dataset_seq/train/static/lat_hr_full.npy",
        #     "output": "/home/yc/Task1/data/v_bar/lat.npy"
        # },
        # {
        #     "description": "元数据 - Longitude",
        #     "input": "/data_new/OceanSR/split_ocean_dataset_seq/train/static/lon_hr_full.npy",
        #     "output": "/home/yc/Task1/data/v_bar/lon.npy"
        # },
        # --- 如果您还想处理验证集和测试集，请取消下面的注释并修改路径 ---
        # {
        #     "description": "高分辨率验证数据",
        #     "input": "/data_new/OceanSR/split_ocean_dataset_seq/val/hr/vbar_hr_val.npy",
        #     "output": "/home/yc/Task1/data/v_bar/vbar_hr_val.npy"
        # },
        # {
        #     "description": "低分辨率验证数据",
        #     "input": "/data_new/OceanSR/split_ocean_dataset_seq/val/lr/vbar_sr_input_val.npy",
        #     "output": "/home/yc/Task1/data/v_bar/vbar_sr_input_val.npy"
        # },
        
        # {
        #     "description": "高分辨率测试数据",
        #     "input": "/data_new/OceanSR/split_ocean_dataset_seq/test/lr/vbar_sr_input_test.npy",
        #     "output": "/home/yc/Task1/data/v_bar/vbar_hr_test.npy"
        # },
        # {
        #     "description": "低分辨率测试数据",
        #     "input": "/data_new/OceanSR/split_ocean_dataset_seq/test/hr/vbar_hr_test.npy",
        #     "output": "/home/yc/Task1/data/v_bar/vbar_sr_input_test.npy"
        # },
    ]
    # ===================================================================

    print("===== 开始创建Demo数据集 =====")
    print(f"将为每个数据集截取 {DEMO_SAMPLE_COUNT} 个样本。\n")
    
    # 循环处理文件列表中的每一个文件
    for item in file_processing_list:
        create_demo_subset(
            input_path=item["input"],
            output_path=item["output"],
            num_samples=DEMO_SAMPLE_COUNT
        )
        
    print("===== 所有任务完成！ =====")