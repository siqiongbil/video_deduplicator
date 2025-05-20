# 视频文件去重工具 | Video Deduplication Tool

这是一个用于检测和删除重复视频文件的Python工具。该工具使用视频帧比较来识别重复视频，并提供交互式的删除确认机制。

A Python tool for detecting and removing duplicate video files. This tool uses video frame comparison to identify duplicate videos and provides an interactive deletion confirmation mechanism.

## 功能特点 | Features

- 递归扫描指定目录及其子目录
- 支持多种视频格式（mp4, avi, mkv, mov, wmv, flv, webm）
- 使用视频帧比较进行重复检测
- 智能提取关键帧（基于场景变化）
- 保存对比帧供用户确认
- 提供多种删除模式
- 详细的日志记录

- Recursively scan specified directories and subdirectories
- Support multiple video formats (mp4, avi, mkv, mov, wmv, flv, webm)
- Use video frame comparison for duplicate detection
- Intelligent key frame extraction (based on scene changes)
- Save comparison frames for user confirmation
- Multiple deletion modes
- Detailed logging

## 安装依赖 | Installation

```bash
pip install opencv-python numpy pillow imagehash
```

## 使用方法 | Usage

1. 基本扫描（预览模式）| Basic scan (preview mode):
```bash
python video_deduplicator.py /path/to/videos
```

2. 执行删除操作 | Execute deletion:
```bash
python video_deduplicator.py /path/to/videos --dry-run false
```

3. 自定义参数 | Custom parameters:
```bash
python video_deduplicator.py /path/to/videos --frames 10 --threshold 3
```

## 参数说明 | Parameters

- `directory`: 要扫描的目录路径（必需）| Directory path to scan (required)
- `--dry-run`: 预览模式，不实际删除文件（默认：True）| Preview mode, no actual deletion (default: True)
- `--frames`: 每个视频提取的帧数（默认：5）| Number of frames to extract per video (default: 5)
- `--threshold`: 视频相似度阈值（默认：5）| Video similarity threshold (default: 5)

## 操作流程 | Operation Process

1. 程序会在指定目录下创建时间戳命名的输出目录（如：`duplicate_frames_20240320_123456`）

2. 对于每组可能的重复视频：
   - 提取关键帧（基于场景变化）
   - 保存对比帧到输出目录
   - 显示视频信息（文件名、时长、大小）

3. 用户可以选择以下操作模式：
   - 逐个确认删除：对每个重复文件单独确认
   - 批量删除所有：一次性删除所有重复文件
   - 退出不删除：取消所有删除操作

4. 删除操作：
   - 删除重复视频文件
   - 自动清理对应的对比帧目录
   - 如果输出目录为空，自动删除

1. The program creates a timestamp-named output directory (e.g., `duplicate_frames_20240320_123456`)

2. For each potential duplicate video group:
   - Extract key frames (based on scene changes)
   - Save comparison frames to output directory
   - Display video information (filename, duration, size)

3. Users can choose from the following operation modes:
   - Confirm deletion one by one: Confirm each duplicate file individually
   - Batch delete all: Delete all duplicate files at once
   - Exit without deletion: Cancel all deletion operations

4. Deletion process:
   - Delete duplicate video files
   - Automatically clean up corresponding comparison frame directories
   - Automatically delete if output directory is empty

## 输出说明 | Output Description

程序会在指定目录下创建以下结构：
```
duplicate_frames_YYYYMMDD_HHMMSS/
├── video1_vs_video2/
│   ├── frame_00.jpg
│   ├── frame_01.jpg
│   └── ...
├── video3_vs_video4/
│   ├── frame_00.jpg
│   ├── frame_01.jpg
│   └── ...
└── ...
```

每个对比帧包含：
- 两个视频的并排显示
- 视频文件名
- 视频时长
- 文件大小
- 帧序号

The program creates the following structure in the specified directory:
```
duplicate_frames_YYYYMMDD_HHMMSS/
├── video1_vs_video2/
│   ├── frame_00.jpg
│   ├── frame_01.jpg
│   └── ...
├── video3_vs_video4/
│   ├── frame_00.jpg
│   ├── frame_01.jpg
│   └── ...
└── ...
```

Each comparison frame includes:
- Side-by-side display of two videos
- Video filename
- Video duration
- File size
- Frame number

## 注意事项 | Notes

1. 确保有足够的磁盘空间用于保存对比帧
2. 删除操作不可恢复，请谨慎操作
3. 建议先使用预览模式（--dry-run）检查结果
4. 程序会自动处理中文文件名和特殊字符
5. 对比帧目录会在以下情况自动删除：
   - 用户选择不删除视频
   - 删除操作完成后
   - 输出目录为空时

1. Ensure sufficient disk space for saving comparison frames
2. Deletion operations cannot be undone, please proceed with caution
3. It's recommended to use preview mode (--dry-run) first
4. The program automatically handles Chinese filenames and special characters
5. Comparison frame directories are automatically deleted when:
   - User chooses not to delete videos
   - Deletion operation is completed
   - Output directory is empty

## 错误处理 | Error Handling

- 程序会记录详细的错误日志
- 单个文件处理失败不会影响整体运行
- 自动清理失败的操作残留文件

- The program logs detailed error information
- Individual file processing failures don't affect overall operation
- Automatically cleans up failed operation residues

## 性能说明 | Performance

- 使用多线程处理视频文件
- 智能提取关键帧减少处理时间
- 优化图像处理提高比较效率

- Uses multi-threading for video file processing
- Intelligent key frame extraction reduces processing time
- Optimized image processing improves comparison efficiency

## 支持的视频格式 | Supported Video Formats

- .mp4
- .avi
- .mkv
- .mov
- .wmv
- .flv
- .webm 