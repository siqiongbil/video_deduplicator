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

def find_duplicates(video_files: List[str]) -> Dict[str, List[str]]:
    """查找重复的视频文件"""
    video_hashes = {}
    duplicates = {}
    
    # 计算所有视频的哈希值
    with ThreadPoolExecutor() as executor:
        future_to_video = {executor.submit(calculate_video_hash, video): video 
                          for video in video_files}
        
        for future in future_to_video:
            video = future_to_video[future]
            try:
                hash_list = future.result()
                if hash_list:
                    video_hashes[video] = hash_list
            except Exception as e:
                logger.error(f"处理视频 {video} 时出错: {str(e)}")
    
    # 查找重复视频
    processed = set()
    for video1, hash_list1 in video_hashes.items():
        if video1 in processed:
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
    
    return duplicates

def process_duplicates(duplicates: Dict[str, List[str]], base_dir: str, dry_run: bool = True) -> None:
    """处理重复文件，保存对比帧并询问用户是否删除"""
    output_dir = create_output_dir(base_dir)
    logger.info(f"对比帧将保存到: {output_dir}")
    
    # 收集所有重复视频组的信息
    duplicate_groups = []
    for keep_file, file_list in duplicates.items():
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
            for group in duplicate_groups:
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
        
        elif choice == "2":
            # 批量删除所有
            confirm = input("\n确定要删除所有重复文件吗？此操作不可恢复！(y/n): ").lower()
            if confirm == 'y':
                for group in duplicate_groups:
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
            else:
                logger.info("已取消批量删除操作")
                # 删除所有对比帧目录
                for group in duplicate_groups:
                    try:
                        shutil.rmtree(group['comparison_dir'])
                        logger.info(f"已删除对比帧目录: {group['comparison_dir']}")
                    except Exception as e:
                        logger.error(f"删除对比帧目录 {group['comparison_dir']} 时出错: {str(e)}")
        
        else:
            logger.info("已取消所有删除操作")
            # 删除所有对比帧目录
            for group in duplicate_groups:
                try:
                    shutil.rmtree(group['comparison_dir'])
                    logger.info(f"已删除对比帧目录: {group['comparison_dir']}")
                except Exception as e:
                    logger.error(f"删除对比帧目录 {group['comparison_dir']} 时出错: {str(e)}")
    
    # 如果输出目录为空，则删除它
    if not dry_run and os.path.exists(output_dir) and not os.listdir(output_dir):
        try:
            os.rmdir(output_dir)
            logger.info(f"已删除空输出目录: {output_dir}")
        except Exception as e:
            logger.error(f"删除空输出目录 {output_dir} 时出错: {str(e)}")

def main():
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

    duplicates = find_duplicates(video_files)
    if not duplicates:
        logger.info("未发现重复文件")
        return

    total_duplicates = sum(len(files) - 1 for files in duplicates.values())
    logger.info(f"发现 {len(duplicates)} 组重复文件，共 {total_duplicates} 个重复文件")
    
    process_duplicates(duplicates, args.directory, args.dry_run)

if __name__ == '__main__':
    main() 