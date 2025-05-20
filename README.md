# 视频去重工具 (Video Deduplicator)

一个基于 Python 的视频文件去重工具，支持通过视频帧对比来识别重复视频。

## 功能特点

- 支持多种视频格式（mp4, avi, mkv, mov, wmv, flv, webm）
- 使用视频帧感知哈希对比，更准确地识别重复视频
- 提供图形界面和命令行两种使用模式
- 支持批量处理和交互式确认
- 自动保存对比帧，方便人工确认
- 支持 Windows 和 Linux 系统

## 安装说明

### 方法一：直接下载可执行文件

1. 从 [Releases]((https://github.com/siqiongbil/video_deduplicator/releases)) 页面下载最新版本
2. 解压下载的文件
3. 直接运行 `video_deduplicator.exe`（Windows）或 `video_deduplicator`（Linux）

### 方法二：从源码安装

1. 克隆仓库：
```bash
git clone https://github.com/siqiongbil/video_deduplicator.git
cd video-deduplicator
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 运行程序：
```bash
python video_deduplicator.py
```

## 使用方法

### 图形界面模式

1. 直接运行程序，打开图形界面
2. 点击"浏览"选择要扫描的视频目录
3. 设置参数：
   - 每个视频提取的帧数（默认：5）
   - 相似度阈值（默认：5）
   - 是否开启预览模式
4. 点击"开始处理"开始扫描
5. 查看对比帧，选择要删除的文件
6. 使用"删除选中文件"或"删除所有文件"进行删除

### 命令行模式

```bash
python video_deduplicator.py <目录路径> [选项]
```

选项：
- `--dry-run`：仅显示将要删除的文件，不实际删除
- `--frames <数量>`：每个视频提取的帧数（默认：5）
- `--threshold <数值>`：视频相似度阈值（默认：5）

示例：
```bash
python video_deduplicator.py ./videos --frames 10 --threshold 3
```

## 参数说明

- **帧数**：每个视频提取的关键帧数量，建议值 5-10
- **相似度阈值**：判断视频相似的阈值，值越小要求越严格
- **预览模式**：开启后只显示重复文件，不实际删除

## 注意事项

1. 程序会在视频目录下创建 `duplicate_frames_时间戳` 目录存放对比帧
2. 删除视频时会同时删除对应的对比帧目录
3. 建议先使用预览模式确认重复文件
4. 大量视频处理可能需要较长时间
5. 请确保有足够的磁盘空间存储对比帧

## 常见问题

1. **Q: 为什么有些明显重复的视频没有被识别？**  
   A: 可以尝试增加帧数或降低相似度阈值

2. **Q: 程序运行很慢怎么办？**  
   A: 减少帧数可以提高处理速度，但可能影响准确性

3. **Q: 对比帧目录可以删除吗？**  
   A: 可以，程序会在删除视频时自动清理对应的对比帧目录

## 开发说明

### 依赖项

- Python 3.6+
- OpenCV (cv2)
- NumPy
- Pillow
- imagehash
- tqdm
- tkinter

### 构建可执行文件

```bash
pyinstaller video_deduplicator.spec
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v1.0.0
- 初始版本发布
- 支持图形界面和命令行模式
- 实现视频帧对比功能
- 支持批量处理和交互确认 
