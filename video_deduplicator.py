#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
import imagehash
from PIL import Image
import shutil
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sys
import threading
import queue
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 支持的视频文件扩展名
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}

def create_output_dir(base_dir: str) -> str:
    """创建输出目录"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"duplicate_frames_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_video_info(video_path: str) -> dict:
    """获取视频信息"""
    info = {
        'duration': 0,
        'size': 0,
        'filename': os.path.basename(video_path)
    }
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            # 获取视频时长（秒）
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info['duration'] = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # 获取文件大小（MB）
        info['size'] = os.path.getsize(video_path) / (1024 * 1024)
    except Exception as e:
        logger.error(f"获取视频信息时出错: {str(e)}")
    
    return info

def format_duration(seconds: float) -> str:
    """格式化时长"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def sanitize_filename(filename: str) -> str:
    """处理文件名，移除或替换不安全的字符"""
    # 移除或替换不安全的字符
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # 处理中文文件名
    try:
        # 使用更安全的编码方式
        filename = filename.encode('utf-8').decode('utf-8')
        # 替换所有非ASCII字符为下划线
        filename = ''.join(c if ord(c) < 128 else '_' for c in filename)
    except UnicodeError:
        # 如果失败，使用更安全的编码方式
        filename = filename.encode('ascii', 'ignore').decode('ascii')
    
    return filename

def save_comparison_frames(video1: str, video2: str, output_dir: str) -> str:
    """保存两个视频的对比帧"""
    try:
        # 获取视频信息
        info1 = get_video_info(video1)
        info2 = get_video_info(video2)
        
        # 创建以视频文件名命名的子目录，处理文件名
        video1_name = sanitize_filename(os.path.splitext(os.path.basename(video1))[0])
        video2_name = sanitize_filename(os.path.splitext(os.path.basename(video2))[0])
        comparison_dir = os.path.join(output_dir, f"{video1_name}_vs_{video2_name}")
        
        # 确保目录路径是合法的
        comparison_dir = os.path.normpath(comparison_dir)
        os.makedirs(comparison_dir, exist_ok=True)
        
        # 提取并保存帧
        frames1 = extract_frames(video1)
        frames2 = extract_frames(video2)
        
        if not frames1 or not frames2:
            logger.error(f"无法提取视频帧: {video1} 或 {video2}")
            return comparison_dir
        
        # 确保两个视频的帧数相同
        min_frames = min(len(frames1), len(frames2))
        frames1 = frames1[:min_frames]
        frames2 = frames2[:min_frames]
        
        for i, (frame1, frame2) in enumerate(zip(frames1, frames2)):
            try:
                # 调整帧大小以便并排显示
                height = max(frame1.shape[0], frame2.shape[0])
                width = max(frame1.shape[1], frame2.shape[1])
                
                # 创建并排显示的图像
                comparison = np.zeros((height + 60, width * 2, 3), dtype=np.uint8)
                
                # 确保帧是BGR格式
                if len(frame1.shape) == 2:  # 如果是灰度图
                    frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
                if len(frame2.shape) == 2:  # 如果是灰度图
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)
                
                # 复制帧到比较图像
                comparison[:frame1.shape[0], :frame1.shape[1]] = frame1
                comparison[:frame2.shape[0], width:width + frame2.shape[1]] = frame2
                
                # 添加视频信息
                info_text1 = f"{info1['filename']} | {format_duration(info1['duration'])} | {info1['size']:.1f}MB"
                info_text2 = f"{info2['filename']} | {format_duration(info2['duration'])} | {info2['size']:.1f}MB"
                
                # 添加标签
                cv2.putText(comparison, info_text1, (10, height + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(comparison, info_text2, (width + 10, height + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 添加帧序号
                cv2.putText(comparison, f"Frame {i+1}", (10, height + 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 保存对比图像
                output_path = os.path.join(comparison_dir, f"frame_{i:02d}.jpg")
                output_path = os.path.normpath(output_path)
                
                # 确保目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # 尝试保存图像
                success = cv2.imwrite(output_path, comparison)
                if not success:
                    logger.error(f"保存对比帧失败: {output_path}")
                    # 尝试使用绝对路径
                    abs_path = os.path.abspath(output_path)
                    success = cv2.imwrite(abs_path, comparison)
                    if not success:
                        logger.error(f"使用绝对路径保存对比帧也失败: {abs_path}")
                
            except Exception as e:
                logger.error(f"处理第 {i+1} 帧时出错: {str(e)}")
                continue
        
        return comparison_dir
        
    except Exception as e:
        logger.error(f"保存对比帧时出错: {str(e)}")
        return output_dir

def extract_frames(video_path: str, num_frames: int = 5) -> List[np.ndarray]:
    """从视频中提取关键帧"""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return frames

        # 获取视频总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return frames

        # 计算采样间隔
        interval = max(1, total_frames // num_frames)
        
        # 提取帧并计算帧差
        prev_frame = None
        frame_diffs = []
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            if prev_frame is not None:
                # 计算帧差
                diff = cv2.absdiff(frame, prev_frame)
                diff_sum = np.sum(diff)
                frame_diffs.append((i, diff_sum))
            
            prev_frame = frame.copy()
        
        # 根据帧差选择关键帧
        if frame_diffs:
            # 按帧差大小排序
            frame_diffs.sort(key=lambda x: x[1], reverse=True)
            # 选择帧差最大的几个位置
            key_positions = [pos for pos, _ in frame_diffs[:num_frames]]
            key_positions.sort()  # 按时间顺序排序
            
            # 重新读取视频并提取关键帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                if i in key_positions:
                    frames.append(frame)
        
        cap.release()
    except Exception as e:
        logger.error(f"处理视频 {video_path} 时出错: {str(e)}")
    
    return frames

def calculate_frame_hash(frame: np.ndarray) -> str:
    """计算帧的感知哈希值"""
    try:
        # 调整图像大小以加快处理速度
        resized = cv2.resize(frame, (32, 32))
        # 转换为灰度图
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # 转换为PIL图像
        pil_image = Image.fromarray(gray)
        # 计算感知哈希
        hash_value = str(imagehash.average_hash(pil_image))
        return hash_value
    except Exception as e:
        logger.error(f"计算帧哈希值时出错: {str(e)}")
        return ""

def calculate_video_hash(video_path: str) -> List[str]:
    """计算视频的帧哈希值列表"""
    frames = extract_frames(video_path)
    return [calculate_frame_hash(frame) for frame in frames if frame is not None]

def are_videos_similar(hash_list1: List[str], hash_list2: List[str], threshold: int = 5) -> bool:
    """比较两个视频的相似度"""
    if not hash_list1 or not hash_list2:
        return False
    
    # 计算哈希值之间的汉明距离
    total_distance = 0
    match_count = 0
    
    for hash1 in hash_list1:
        min_distance = float('inf')
        for hash2 in hash_list2:
            distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
            min_distance = min(min_distance, distance)
        
        if min_distance <= threshold:
            match_count += 1
            total_distance += min_distance
    
    # 计算匹配率
    match_ratio = match_count / len(hash_list1)
    # 计算平均距离
    avg_distance = total_distance / match_count if match_count > 0 else float('inf')
    
    # 要求至少50%的帧匹配，且平均距离小于阈值
    return match_ratio >= 0.5 and avg_distance <= threshold

def find_video_files(directory: str) -> List[str]:
    """递归查找目录中的所有视频文件"""
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in VIDEO_EXTENSIONS:
                video_files.append(os.path.join(root, file))
    return video_files

def find_duplicates(video_files: List[str], progress_callback=None, show_progress=False) -> Dict[str, List[str]]:
    """查找重复的视频文件"""
    video_hashes = {}
    duplicates = {}
    
    # 计算所有视频的哈希值
    with ThreadPoolExecutor() as executor:
        future_to_video = {executor.submit(calculate_video_hash, video): video 
                          for video in video_files}
        
        total = len(video_files)
        for i, future in enumerate(future_to_video):
            video = future_to_video[future]
            try:
                hash_list = future.result()
                if hash_list:
                    video_hashes[video] = hash_list
            except Exception as e:
                logger.error(f"处理视频 {video} 时出错: {str(e)}")
            finally:
                if progress_callback:
                    progress_callback(i + 1, total, "分析视频")
                elif show_progress:
                    logger.info(f"正在分析视频... ({i + 1}/{total})")
    
    # 查找重复视频
    processed = set()
    total = len(video_hashes)
    for i, (video1, hash_list1) in enumerate(video_hashes.items()):
        if video1 in processed:
            if progress_callback:
                progress_callback(i + 1, total, "查找重复")
            elif show_progress:
                logger.info(f"正在查找重复... ({i + 1}/{total})")
            continue
            
        similar_videos = [video1]
        for video2, hash_list2 in video_hashes.items():
            if video1 != video2 and video2 not in processed:
                if are_videos_similar(hash_list1, hash_list2):
                    similar_videos.append(video2)
                    processed.add(video2)
        
        if len(similar_videos) > 1:
            duplicates[video1] = similar_videos
            processed.add(video1)
        
        if progress_callback:
            progress_callback(i + 1, total, "查找重复")
        elif show_progress:
            logger.info(f"正在查找重复... ({i + 1}/{total})")
    
    return duplicates

def process_duplicates(duplicates: Dict[str, List[str]], base_dir: str, dry_run: bool = True, progress_callback=None, show_progress=False) -> None:
    """处理重复文件，保存对比帧并询问用户是否删除"""
    output_dir = create_output_dir(base_dir)
    logger.info(f"对比帧将保存到: {output_dir}")
    
    # 收集所有重复视频组的信息
    duplicate_groups = []
    total_groups = len(duplicates)
    
    for i, (keep_file, file_list) in enumerate(duplicates.items(), 1):
        delete_files = file_list[1:]
        for file_to_delete in delete_files:
            comparison_dir = save_comparison_frames(keep_file, file_to_delete, output_dir)
            logger.info(f"已保存对比帧到: {comparison_dir}")
            
            # 获取视频信息
            keep_info = get_video_info(keep_file)
            delete_info = get_video_info(file_to_delete)
            
            duplicate_groups.append({
                'keep_file': keep_file,
                'delete_file': file_to_delete,
                'comparison_dir': comparison_dir,
                'keep_info': keep_info,
                'delete_info': delete_info
            })
        
        if progress_callback:
            progress_callback(i, total_groups, "保存对比帧")
        elif show_progress:
            logger.info(f"正在保存对比帧... ({i}/{total_groups})")
    
    if not dry_run and duplicate_groups:
        # 显示所有重复视频组的信息
        logger.info("\n发现以下可能的重复视频组：")
        for i, group in enumerate(duplicate_groups, 1):
            logger.info(f"\n组 {i}:")
            logger.info(f"视频1: {group['keep_file']}")
            logger.info(f"  - 时长: {format_duration(group['keep_info']['duration'])}")
            logger.info(f"  - 大小: {group['keep_info']['size']:.1f}MB")
            logger.info(f"视频2: {group['delete_file']}")
            logger.info(f"  - 时长: {format_duration(group['delete_info']['duration'])}")
            logger.info(f"  - 大小: {group['delete_info']['size']:.1f}MB")
            logger.info(f"对比帧: {group['comparison_dir']}")
        
        logger.info("\n请查看对比帧，删除不重复视频的对比帧目录")
        logger.info("然后选择操作模式：")
        logger.info("1. 逐个确认删除")
        logger.info("2. 批量删除所有")
        logger.info("3. 退出不删除")
        
        choice = input("请输入选项 (1/2/3): ").strip()
        
        if choice == "1":
            # 逐个确认删除
            if progress_callback:
                pbar = tqdm(total=len(duplicate_groups), desc="删除文件", unit="个")
            elif show_progress:
                logger.info(f"开始逐个删除文件... (共{len(duplicate_groups)}个)")
            
            for i, group in enumerate(duplicate_groups, 1):
                if os.path.exists(group['comparison_dir']):
                    response = input(f"\n是否删除文件 {group['delete_file']}? (y/n): ").lower()
                    if response == 'y':
                        try:
                            os.remove(group['delete_file'])
                            logger.info(f"已删除: {group['delete_file']}")
                            # 删除对应的对比帧目录
                            try:
                                shutil.rmtree(group['comparison_dir'])
                                logger.info(f"已删除对比帧目录: {group['comparison_dir']}")
                            except Exception as e:
                                logger.error(f"删除对比帧目录 {group['comparison_dir']} 时出错: {str(e)}")
                        except Exception as e:
                            logger.error(f"删除文件 {group['delete_file']} 时出错: {str(e)}")
                    else:
                        logger.info(f"保留文件: {group['delete_file']}")
                        # 删除对应的对比帧目录
                        try:
                            shutil.rmtree(group['comparison_dir'])
                            logger.info(f"已删除对比帧目录: {group['comparison_dir']}")
                        except Exception as e:
                            logger.error(f"删除对比帧目录 {group['comparison_dir']} 时出错: {str(e)}")
                
                if progress_callback:
                    pbar.update(1)
                elif show_progress:
                    logger.info(f"已处理 {i}/{len(duplicate_groups)} 个文件")
            
            if progress_callback:
                pbar.close()
        
        elif choice == "2":
            # 批量删除所有
            confirm = input("\n确定要删除所有重复文件吗？此操作不可恢复！(y/n): ").lower()
            if confirm == 'y':
                if progress_callback:
                    pbar = tqdm(total=len(duplicate_groups), desc="删除文件", unit="个")
                elif show_progress:
                    logger.info(f"开始批量删除文件... (共{len(duplicate_groups)}个)")
                
                for i, group in enumerate(duplicate_groups, 1):
                    if os.path.exists(group['comparison_dir']):
                        try:
                            os.remove(group['delete_file'])
                            logger.info(f"已删除: {group['delete_file']}")
                            # 删除对应的对比帧目录
                            try:
                                shutil.rmtree(group['comparison_dir'])
                                logger.info(f"已删除对比帧目录: {group['comparison_dir']}")
                            except Exception as e:
                                logger.error(f"删除对比帧目录 {group['comparison_dir']} 时出错: {str(e)}")
                        except Exception as e:
                            logger.error(f"删除文件 {group['delete_file']} 时出错: {str(e)}")
                    
                    if progress_callback:
                        pbar.update(1)
                    elif show_progress:
                        logger.info(f"已处理 {i}/{len(duplicate_groups)} 个文件")
                
                if progress_callback:
                    pbar.close()
            else:
                logger.info("已取消批量删除操作")
                # 删除所有对比帧目录
                if progress_callback:
                    pbar = tqdm(total=len(duplicate_groups), desc="清理对比帧", unit="个")
                elif show_progress:
                    logger.info(f"开始清理对比帧... (共{len(duplicate_groups)}个)")
                
                for i, group in enumerate(duplicate_groups, 1):
                    try:
                        shutil.rmtree(group['comparison_dir'])
                        logger.info(f"已删除对比帧目录: {group['comparison_dir']}")
                    except Exception as e:
                        logger.error(f"删除对比帧目录 {group['comparison_dir']} 时出错: {str(e)}")
                    
                    if progress_callback:
                        pbar.update(1)
                    elif show_progress:
                        logger.info(f"已处理 {i}/{len(duplicate_groups)} 个文件")
                
                if progress_callback:
                    pbar.close()
        
        else:
            logger.info("已取消所有删除操作")
            # 删除所有对比帧目录
            if progress_callback:
                pbar = tqdm(total=len(duplicate_groups), desc="清理对比帧", unit="个")
            elif show_progress:
                logger.info(f"开始清理对比帧... (共{len(duplicate_groups)}个)")
            
            for i, group in enumerate(duplicate_groups, 1):
                try:
                    shutil.rmtree(group['comparison_dir'])
                    logger.info(f"已删除对比帧目录: {group['comparison_dir']}")
                except Exception as e:
                    logger.error(f"删除对比帧目录 {group['comparison_dir']} 时出错: {str(e)}")
                
                if progress_callback:
                    pbar.update(1)
                elif show_progress:
                    logger.info(f"已处理 {i}/{len(duplicate_groups)} 个文件")
            
            if progress_callback:
                pbar.close()
    
    # 如果输出目录为空，则删除它
    if not dry_run and os.path.exists(output_dir) and not os.listdir(output_dir):
        try:
            os.rmdir(output_dir)
            logger.info(f"已删除空输出目录: {output_dir}")
        except Exception as e:
            logger.error(f"删除空输出目录 {output_dir} 时出错: {str(e)}")

class VideoDeduplicatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("视频去重工具 | Video Deduplication Tool")
        self.root.geometry("1000x800")  # 增加窗口大小
        
        # 设置窗口图标
        try:
            # 获取应用程序路径
            if getattr(sys, 'frozen', False):
                # 如果是打包后的可执行文件
                application_path = sys._MEIPASS
            else:
                # 如果是开发环境
                application_path = os.path.dirname(os.path.abspath(__file__))
            
            icon_path = os.path.join(application_path, 'image', 'favicons.ico')
            
            if os.path.exists(icon_path):
                # 确保图标文件存在
                self.root.iconbitmap(icon_path)  # 移除 default= 参数
                logger.info(f"成功设置窗口图标: {icon_path}")
            else:
                logger.error(f"图标文件不存在: {icon_path}")
        except Exception as e:
            logger.error(f"设置窗口图标时出错: {str(e)}")
        
        # 创建消息队列用于线程间通信
        self.message_queue = queue.Queue()
        
        # 创建主框架
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置根窗口的网格权重，使其可以随窗口调整大小
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        
        # 配置主框架的网格权重
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(3, weight=1)  # 让列表区域可以随窗口调整大小
        
        # 目录选择
        ttk.Label(main_frame, text="选择视频目录:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.dir_path = tk.StringVar()
        self.dir_entry = ttk.Entry(main_frame, textvariable=self.dir_path, width=50)
        self.dir_entry.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        self.browse_button = ttk.Button(main_frame, text="浏览", command=self.browse_directory)
        self.browse_button.grid(row=0, column=2, padx=5)
        
        # 参数设置
        param_frame = ttk.LabelFrame(main_frame, text="参数设置", padding="5")
        param_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        param_frame.grid_columnconfigure(1, weight=1)
        
        # 帧数设置
        ttk.Label(param_frame, text="每个视频提取的帧数:").grid(row=0, column=0, sticky=tk.W)
        self.frames = tk.StringVar(value="5")
        self.frames_entry = ttk.Entry(param_frame, textvariable=self.frames, width=10)
        self.frames_entry.grid(row=0, column=1, padx=5, sticky=tk.W)
        
        # 阈值设置
        ttk.Label(param_frame, text="相似度阈值:").grid(row=1, column=0, sticky=tk.W)
        self.threshold = tk.StringVar(value="5")
        self.threshold_entry = ttk.Entry(param_frame, textvariable=self.threshold, width=10)
        self.threshold_entry.grid(row=1, column=1, padx=5, sticky=tk.W)
        
        # 预览模式
        self.dry_run = tk.BooleanVar(value=True)
        self.dry_run_check = ttk.Checkbutton(param_frame, text="预览模式（不实际删除）", variable=self.dry_run)
        self.dry_run_check.grid(row=2, column=0, columnspan=2, sticky=tk.W)
        
        # 进度显示框架
        progress_frame = ttk.LabelFrame(main_frame, text="处理进度", padding="5")
        progress_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        progress_frame.grid_columnconfigure(1, weight=1)
        
        # 总体进度
        ttk.Label(progress_frame, text="总体进度:").grid(row=0, column=0, sticky=tk.W)
        self.total_progress = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.total_progress.grid(row=0, column=1, padx=5, pady=2, sticky=(tk.W, tk.E))
        self.total_progress_label = ttk.Label(progress_frame, text="0%")
        self.total_progress_label.grid(row=0, column=2, padx=5)
        
        # 当前阶段进度
        ttk.Label(progress_frame, text="当前阶段:").grid(row=1, column=0, sticky=tk.W)
        self.stage_progress = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.stage_progress.grid(row=1, column=1, padx=5, pady=2, sticky=(tk.W, tk.E))
        self.stage_progress_label = ttk.Label(progress_frame, text="0%")
        self.stage_progress_label.grid(row=1, column=2, padx=5)
        
        # 状态标签
        self.status_label = ttk.Label(progress_frame, text="就绪")
        self.status_label.grid(row=2, column=0, columnspan=3, pady=2)
        
        # 创建上下分割的框架
        paned_frame = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        paned_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # 重复文件列表框架
        list_frame = ttk.LabelFrame(paned_frame, text="重复文件列表", padding="5")
        paned_frame.add(list_frame, weight=2)
        
        # 创建Treeview
        columns = ('选择', '保留文件', '删除文件', '时长', '大小', '对比帧目录')
        self.tree = ttk.Treeview(list_frame, columns=columns, show='headings', selectmode='extended')
        
        # 设置列标题和宽度
        self.tree.heading('选择', text='选择')
        self.tree.column('选择', width=50, anchor='center')
        for col in columns[1:]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 绑定双击事件和点击事件
        self.tree.bind('<Double-1>', self.on_double_click)
        self.tree.bind('<ButtonRelease-1>', self.on_click)
        
        # 添加全选/取消全选按钮
        select_frame = ttk.Frame(list_frame)
        select_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.select_all_var = tk.BooleanVar(value=False)
        self.select_all_check = ttk.Checkbutton(
            select_frame, 
            text="全选/取消全选",
            variable=self.select_all_var,
            command=self.toggle_select_all
        )
        self.select_all_check.pack(side=tk.LEFT, padx=5)
        
        # 日志显示框架
        log_frame = ttk.LabelFrame(paned_frame, text="处理日志", padding="5")
        paned_frame.add(log_frame, weight=1)
        
        # 日志显示
        self.log_text = tk.Text(log_frame, height=10, width=60)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # 添加滚动条
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        # 操作按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=10)
        
        # 开始按钮
        self.start_button = ttk.Button(button_frame, text="开始处理", command=self.start_processing)
        self.start_button.grid(row=0, column=0, padx=5)
        
        # 删除选中按钮
        self.delete_selected_button = ttk.Button(button_frame, text="删除选中文件", command=self.delete_selected, state='disabled')
        self.delete_selected_button.grid(row=0, column=1, padx=5)
        
        # 删除所有按钮
        self.delete_all_button = ttk.Button(button_frame, text="删除所有文件", command=self.delete_all, state='disabled')
        self.delete_all_button.grid(row=0, column=2, padx=5)
        
        # 配置日志处理器
        self.log_handler = TextHandler(self.log_text)
        logger.addHandler(self.log_handler)
        
        # 启动消息处理
        self.process_messages()
        
        # 进度跟踪变量
        self.total_steps = 0
        self.current_step = 0
        self.current_stage = ""
        self.stage_steps = 0
        self.stage_current = 0
        
        # 存储重复文件信息
        self.duplicate_groups = []
        
        # 设置最小窗口大小
        self.root.update_idletasks()
        min_width = self.root.winfo_reqwidth()
        min_height = self.root.winfo_reqheight()
        self.root.minsize(min_width, min_height)
    
    def toggle_select_all(self):
        """全选/取消全选"""
        for item in self.tree.get_children():
            self.tree.set(item, '选择', '✓' if self.select_all_var.get() else '')
    
    def on_click(self, event):
        """处理点击事件"""
        region = self.tree.identify_region(event.x, event.y)
        if region == "cell":
            column = self.tree.identify_column(event.x)
            if column == '#1':  # 第一列（选择列）
                item = self.tree.identify_row(event.y)
                if item:
                    current = self.tree.set(item, '选择')
                    self.tree.set(item, '选择', '' if current == '✓' else '✓')
    
    def update_duplicate_list(self):
        """更新重复文件列表"""
        # 清空现有项目
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # 添加新的项目
        for group in self.duplicate_groups:
            values = (
                '',  # 选择列初始为空
                os.path.basename(group['keep_file']),
                os.path.basename(group['delete_file']),
                format_duration(group['delete_info']['duration']),
                f"{group['delete_info']['size']:.1f}MB",
                group['comparison_dir']
            )
            self.tree.insert('', 'end', values=values)
    
    def get_selected_items(self):
        """获取选中的项目"""
        selected = []
        for item in self.tree.get_children():
            if self.tree.set(item, '选择') == '✓':
                selected.append(item)
        return selected
    
    def delete_selected(self):
        """删除选中的文件"""
        selected_items = self.get_selected_items()
        if not selected_items:
            messagebox.showinfo("提示", "请先选择要删除的文件")
            return
        
        if not messagebox.askyesno("确认删除", f"确定要删除选中的 {len(selected_items)} 个文件吗？"):
            return
        
        # 禁用控件
        self.set_widgets_state('disabled')
        self.total_progress['value'] = 0
        self.stage_progress['value'] = 0
        self.status_label['text'] = "正在删除..."
        
        # 在新线程中执行删除操作
        thread = threading.Thread(
            target=self.delete_selected_async,
            args=(selected_items,)
        )
        thread.daemon = True
        thread.start()
    
    def delete_selected_async(self, selected_items):
        """异步删除选中的文件"""
        try:
            total = len(selected_items)
            deleted_count = 0
            
            for i, item in enumerate(selected_items, 1):
                values = self.tree.item(item)['values']
                comparison_dir = values[5]  # 对比帧目录现在是第6列
                
                # 查找对应的文件信息
                for group in self.duplicate_groups:
                    if group['comparison_dir'] == comparison_dir:
                        try:
                            os.remove(group['delete_file'])
                            self.message_queue.put({'type': 'log', 'text': f"已删除: {group['delete_file']}"})
                            deleted_count += 1
                        except Exception as e:
                            self.message_queue.put({'type': 'error', 'text': f"删除文件 {group['delete_file']} 时出错: {str(e)}"})
                        
                        # 删除对比帧目录
                        try:
                            shutil.rmtree(comparison_dir)
                            self.message_queue.put({'type': 'log', 'text': f"已删除对比帧目录: {comparison_dir}"})
                        except Exception as e:
                            self.message_queue.put({'type': 'error', 'text': f"删除对比帧目录 {comparison_dir} 时出错: {str(e)}"})
                        
                        # 从列表中移除
                        self.duplicate_groups.remove(group)
                        break
                
                # 更新进度
                progress = (i / total) * 100
                self.update_progress(progress, progress, f"正在删除文件... ({i}/{total})")
            
            # 更新列表显示
            self.message_queue.put({'type': 'update_list'})
            self.message_queue.put({'type': 'info', 'text': f"删除完成，共删除 {deleted_count} 个文件"})
            
        except Exception as e:
            self.message_queue.put({'type': 'error', 'text': f"删除过程中出错：{str(e)}"})
        finally:
            self.message_queue.put({'type': 'done'})
    
    def delete_all(self):
        """删除所有重复文件"""
        if not self.duplicate_groups:
            messagebox.showinfo("提示", "没有需要删除的重复文件")
            return
        
        if not messagebox.askyesno("确认删除", f"确定要删除所有 {len(self.duplicate_groups)} 个重复文件吗？"):
            return
        
        # 禁用控件
        self.set_widgets_state('disabled')
        self.total_progress['value'] = 0
        self.stage_progress['value'] = 0
        self.status_label['text'] = "正在删除..."
        
        # 在新线程中执行删除操作
        thread = threading.Thread(
            target=self.delete_all_async
        )
        thread.daemon = True
        thread.start()
    
    def delete_all_async(self):
        """异步删除所有重复文件"""
        try:
            total = len(self.duplicate_groups)
            deleted_count = 0
            
            for i, group in enumerate(self.duplicate_groups, 1):
                try:
                    os.remove(group['delete_file'])
                    self.message_queue.put({'type': 'log', 'text': f"已删除: {group['delete_file']}"})
                    deleted_count += 1
                except Exception as e:
                    self.message_queue.put({'type': 'error', 'text': f"删除文件 {group['delete_file']} 时出错: {str(e)}"})
                
                # 删除对比帧目录
                try:
                    shutil.rmtree(group['comparison_dir'])
                    self.message_queue.put({'type': 'log', 'text': f"已删除对比帧目录: {group['comparison_dir']}"})
                except Exception as e:
                    self.message_queue.put({'type': 'error', 'text': f"删除对比帧目录 {group['comparison_dir']} 时出错: {str(e)}"})
                
                # 更新进度
                progress = (i / total) * 100
                self.update_progress(progress, progress, f"正在删除文件... ({i}/{total})")
            
            # 清空列表
            self.duplicate_groups = []
            self.message_queue.put({'type': 'update_list'})
            self.message_queue.put({'type': 'info', 'text': f"删除完成，共删除 {deleted_count} 个文件"})
            
        except Exception as e:
            self.message_queue.put({'type': 'error', 'text': f"删除过程中出错：{str(e)}"})
        finally:
            self.message_queue.put({'type': 'done'})
    
    def process_messages(self):
        """处理来自工作线程的消息"""
        try:
            while True:
                message = self.message_queue.get_nowait()
                if message['type'] == 'progress':
                    self.total_progress['value'] = message['total_value']
                    self.total_progress_label['text'] = f"{message['total_value']:.1f}%"
                    self.stage_progress['value'] = message['stage_value']
                    self.stage_progress_label['text'] = f"{message['stage_value']:.1f}%"
                    self.status_label['text'] = message['text']
                elif message['type'] == 'log':
                    logger.info(message['text'])
                elif message['type'] == 'error':
                    messagebox.showerror("错误", message['text'])
                elif message['type'] == 'info':
                    messagebox.showinfo("提示", message['text'])
                elif message['type'] == 'update_list':
                    self.update_duplicate_list()
                elif message['type'] == 'done':
                    self.set_widgets_state('normal')
                    self.total_progress['value'] = 0
                    self.stage_progress['value'] = 0
                    self.total_progress_label['text'] = "0%"
                    self.stage_progress_label['text'] = "0%"
                    self.status_label['text'] = "就绪"
                    if self.duplicate_groups:
                        self.delete_selected_button.configure(state='normal')
                        self.delete_all_button.configure(state='normal')
                    break
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_messages)
    
    def process_videos_async(self, directory, frames, threshold):
        """异步处理视频"""
        try:
            # 初始化进度
            self.total_steps = 0
            self.current_step = 0
            self.stage_steps = 0
            self.stage_current = 0
            self.duplicate_groups = []
            
            # 扫描视频文件
            self.update_progress(0, 0, "正在扫描视频文件...")
            video_files = find_video_files(directory)
            self.message_queue.put({'type': 'log', 'text': f"找到 {len(video_files)} 个视频文件"})
            
            if not video_files:
                self.message_queue.put({'type': 'info', 'text': "未找到视频文件"})
                return
            
            # 分析视频
            self.update_progress(10, 0, "正在分析视频...")
            self.stage_steps = len(video_files)
            self.stage_current = 0
            
            def progress_callback(current, total, stage):
                self.stage_current = current
                stage_progress = (current / total) * 100
                total_progress = 10 + (stage_progress * 0.3)  # 分析阶段占总进度的30%
                self.update_progress(total_progress, stage_progress, f"正在分析视频... ({current}/{total})")
            
            duplicates = find_duplicates(video_files, progress_callback=progress_callback)
            
            if not duplicates:
                self.message_queue.put({'type': 'info', 'text': "未发现重复文件"})
                return
            
            total_duplicates = sum(len(files) - 1 for files in duplicates.values())
            self.message_queue.put({'type': 'log', 'text': f"发现 {len(duplicates)} 组重复文件，共 {total_duplicates} 个重复文件"})
            
            # 保存对比帧
            self.update_progress(40, 0, "正在保存对比帧...")
            self.stage_steps = len(duplicates)
            self.stage_current = 0
            
            def save_progress_callback(current, total, stage):
                self.stage_current = current
                stage_progress = (current / total) * 100
                total_progress = 40 + (stage_progress * 0.4)  # 保存阶段占总进度的40%
                self.update_progress(total_progress, stage_progress, f"正在保存对比帧... ({current}/{total})")
            
            # 收集重复文件信息
            output_dir = create_output_dir(directory)
            for keep_file, file_list in duplicates.items():
                delete_files = file_list[1:]
                for file_to_delete in delete_files:
                    comparison_dir = save_comparison_frames(keep_file, file_to_delete, output_dir)
                    self.message_queue.put({'type': 'log', 'text': f"已保存对比帧到: {comparison_dir}"})
                    
                    # 获取视频信息
                    keep_info = get_video_info(keep_file)
                    delete_info = get_video_info(file_to_delete)
                    
                    self.duplicate_groups.append({
                        'keep_file': keep_file,
                        'delete_file': file_to_delete,
                        'comparison_dir': comparison_dir,
                        'keep_info': keep_info,
                        'delete_info': delete_info
                    })
            
            self.update_progress(100, 100, "处理完成")
            self.message_queue.put({'type': 'update_list'})
            self.message_queue.put({'type': 'info', 'text': "处理完成，请查看对比帧后选择要删除的文件"})
            
        except Exception as e:
            self.message_queue.put({'type': 'error', 'text': f"处理过程中出错：{str(e)}"})
        finally:
            self.message_queue.put({'type': 'done'})
    
    def browse_directory(self):
        """打开目录选择对话框"""
        directory = filedialog.askdirectory()
        if directory:
            self.dir_path.set(directory)
    
    def set_widgets_state(self, state):
        """设置所有控件的状态"""
        self.dir_entry.configure(state=state)
        self.browse_button.configure(state=state)
        self.frames_entry.configure(state=state)
        self.threshold_entry.configure(state=state)
        self.dry_run_check.configure(state=state)
        self.start_button.configure(state=state)
        self.delete_selected_button.configure(state=state)
        self.delete_all_button.configure(state=state)
    
    def update_progress(self, total_value, stage_value, text):
        """更新进度条和状态"""
        self.message_queue.put({
            'type': 'progress',
            'total_value': total_value,
            'stage_value': stage_value,
            'text': text
        })
    
    def start_processing(self):
        """开始处理视频"""
        directory = self.dir_path.get()
        if not directory:
            messagebox.showerror("错误", "请选择视频目录")
            return
        
        try:
            frames = int(self.frames.get())
            threshold = int(self.threshold.get())
        except ValueError:
            messagebox.showerror("错误", "帧数和阈值必须是整数")
            return
        
        # 禁用控件
        self.set_widgets_state('disabled')
        self.total_progress['value'] = 0
        self.stage_progress['value'] = 0
        self.status_label['text'] = "正在处理..."
        
        # 在新线程中处理视频
        thread = threading.Thread(
            target=self.process_videos_async,
            args=(directory, frames, threshold)
        )
        thread.daemon = True
        thread.start()

    def on_double_click(self, event):
        """处理双击事件"""
        item = self.tree.selection()[0]
        comparison_dir = self.tree.item(item)['values'][5]  # 对比帧目录现在是第6列
        if os.path.exists(comparison_dir):
            os.startfile(comparison_dir)  # Windows
            # 对于Linux系统，可以使用：
            # subprocess.run(['xdg-open', comparison_dir])

class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        logging.Handler.__init__(self)
        self.text_widget = text_widget
    
    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)
            self.text_widget.configure(state='disabled')
        self.text_widget.after(0, append)

def main():
    if len(sys.argv) > 1:
        # 命令行模式
        parser = argparse.ArgumentParser(description='视频文件去重工具')
        parser.add_argument('directory', help='要扫描的目录路径')
        parser.add_argument('--dry-run', action='store_true', help='仅显示将要删除的文件，不实际删除')
        parser.add_argument('--frames', type=int, default=5, help='每个视频提取的帧数（默认：5）')
        parser.add_argument('--threshold', type=int, default=5, help='视频相似度阈值（默认：5）')
        args = parser.parse_args()

        if not os.path.isdir(args.directory):
            logger.error(f"指定的路径不是目录: {args.directory}")
            return

        logger.info(f"开始扫描目录: {args.directory}")
        video_files = find_video_files(args.directory)
        logger.info(f"找到 {len(video_files)} 个视频文件")

        duplicates = find_duplicates(video_files, show_progress=True)
        if not duplicates:
            logger.info("未发现重复文件")
            return

        total_duplicates = sum(len(files) - 1 for files in duplicates.values())
        logger.info(f"发现 {len(duplicates)} 组重复文件，共 {total_duplicates} 个重复文件")
        
        process_duplicates(duplicates, args.directory, args.dry_run, show_progress=True)
    else:
        # GUI模式
        try:
            # 检查tkinter是否可用
            import tkinter
            logger.info("正在启动图形界面...")
            
            # 创建根窗口
            root = tk.Tk()
            root.withdraw()  # 暂时隐藏窗口
            
            # 检查是否支持图形界面
            try:
                root.update()
                root.deiconify()  # 显示窗口
            except Exception as e:
                logger.error(f"无法创建图形窗口: {str(e)}")
                raise
            
            # 创建应用实例
            try:
                app = VideoDeduplicatorGUI(root)
                logger.info("图形界面初始化成功")
            except Exception as e:
                logger.error(f"初始化图形界面时出错: {str(e)}")
                raise
            
            # 启动主循环
            try:
                root.mainloop()
            except Exception as e:
                logger.error(f"运行图形界面时出错: {str(e)}")
                raise
                
        except Exception as e:
            # 如果GUI启动失败，尝试使用命令行模式
            logger.error(f"图形界面启动失败: {str(e)}")
            logger.info("尝试使用命令行模式...")
            
            # 获取当前目录
            current_dir = os.path.abspath(".")
            logger.info(f"当前目录: {current_dir}")
            
            # 扫描当前目录
            video_files = find_video_files(current_dir)
            if video_files:
                logger.info(f"找到 {len(video_files)} 个视频文件")
                duplicates = find_duplicates(video_files, show_progress=True)
                if duplicates:
                    total_duplicates = sum(len(files) - 1 for files in duplicates.values())
                    logger.info(f"发现 {len(duplicates)} 组重复文件，共 {total_duplicates} 个重复文件")
                    process_duplicates(duplicates, current_dir, True, show_progress=True)
                else:
                    logger.info("未发现重复文件")
            else:
                logger.info("当前目录下未找到视频文件")
            
            # 显示使用说明
            logger.info("\n使用说明：")
            logger.info("1. 图形界面模式：直接运行程序")
            logger.info("2. 命令行模式：运行程序时指定目录路径")
            logger.info("   例如：python video_deduplicator.py ./videos")
            logger.info("   可选参数：")
            logger.info("   --dry-run：仅显示将要删除的文件，不实际删除")
            logger.info("   --frames：每个视频提取的帧数（默认：5）")
            logger.info("   --threshold：视频相似度阈值（默认：5）")

if __name__ == '__main__':
    main() 