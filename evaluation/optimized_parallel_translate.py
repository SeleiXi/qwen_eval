#!/usr/bin/env python3
"""
Optimized Parallel Video Translation Script
性能优化的并行视频翻译脚本

主要优化：
1. 中间状态保存和断点续传
2. 批处理优化减少GPU切换开销  
3. 资源管理优化
4. 更好的并行策略
5. 实时进度监控

Author: Assistant
Date: 2024
"""

import os
import argparse
import torch
import soundfile as sf
from pathlib import Path
import glob
import subprocess
import tempfile
import shutil
import time
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from queue import Queue, Empty
import multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

@dataclass
class VideoTask:
    """视频处理任务数据结构"""
    video_path: str
    relative_path: str
    output_path: str
    file_hash: str
    file_size: int
    duration: Optional[float] = None
    status: str = "pending"  # pending, processing, completed, failed
    error_msg: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    translation_preview: str = ""
    segments_count: int = 0

@dataclass
class ProcessingStats:
    """处理统计信息"""
    total_files: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    segmented: int = 0
    total_duration: float = 0.0
    processing_time: float = 0.0

class ProgressManager:
    """进度管理器 - 负责保存和恢复处理状态"""
    
    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = checkpoint_file
        self.tasks: Dict[str, VideoTask] = {}
        self.stats = ProcessingStats()
        self.lock = threading.Lock()
        
    def load_checkpoint(self) -> bool:
        """加载检查点文件"""
        if not os.path.exists(self.checkpoint_file):
            return False
            
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 恢复任务状态
            for task_data in data.get('tasks', []):
                task = VideoTask(**task_data)
                self.tasks[task.file_hash] = task
                
            # 恢复统计信息
            stats_data = data.get('stats', {})
            self.stats = ProcessingStats(**stats_data)
            
            print(f"📂 Loaded checkpoint: {len(self.tasks)} tasks, {self.stats.completed} completed")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            return False
    
    def save_checkpoint(self):
        """保存检查点文件"""
        with self.lock:
            try:
                checkpoint_dir = os.path.dirname(self.checkpoint_file)
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                
                data = {
                    'tasks': [asdict(task) for task in self.tasks.values()],
                    'stats': asdict(self.stats),
                    'timestamp': datetime.now().isoformat(),
                    'version': '2.0'
                }
                
                # 原子写入
                temp_file = self.checkpoint_file + '.tmp'
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                if os.path.exists(self.checkpoint_file):
                    os.replace(temp_file, self.checkpoint_file)
                else:
                    os.rename(temp_file, self.checkpoint_file)
                    
            except Exception as e:
                print(f"⚠️ Failed to save checkpoint: {e}")
    
    def add_task(self, task: VideoTask):
        """添加任务"""
        with self.lock:
            self.tasks[task.file_hash] = task
            self.stats.total_files += 1
    
    def update_task(self, file_hash: str, **updates):
        """更新任务状态"""
        with self.lock:
            if file_hash in self.tasks:
                task = self.tasks[file_hash]
                old_status = task.status
                
                for key, value in updates.items():
                    setattr(task, key, value)
                
                # 更新统计信息
                if 'status' in updates:
                    new_status = updates['status']
                    if old_status != 'completed' and new_status == 'completed':
                        self.stats.completed += 1
                    elif old_status != 'failed' and new_status == 'failed':
                        self.stats.failed += 1
    
    def get_pending_tasks(self) -> List[VideoTask]:
        """获取待处理任务"""
        with self.lock:
            return [task for task in self.tasks.values() 
                   if task.status in ['pending', 'failed']]
    
    def get_progress_summary(self) -> Dict:
        """获取进度摘要"""
        with self.lock:
            total = len(self.tasks)
            if total == 0:
                return {'progress': 0, 'completed': 0, 'total': 0}
            
            completed = sum(1 for task in self.tasks.values() if task.status == 'completed')
            failed = sum(1 for task in self.tasks.values() if task.status == 'failed')
            processing = sum(1 for task in self.tasks.values() if task.status == 'processing')
            
            return {
                'progress': (completed / total) * 100,
                'completed': completed,
                'failed': failed,
                'processing': processing,
                'total': total,
                'remaining': total - completed - failed
            }

# 全局模型管理 (借鉴batch版本的高效策略)
_shared_model = None
_shared_processor = None
_model_lock = threading.Lock()
_gpu_semaphore = None

def get_shared_model(model_path: str, use_flash_attn: bool = False, save_audio: bool = False):
    """
    获取共享模型实例（线程安全） - 采用batch版本的高效策略
    """
    global _shared_model, _shared_processor
    
    with _model_lock:
        if _shared_model is None:
            print(f"🤖 Loading shared model from {model_path}...")
            start_time = time.time()
            
            model_kwargs = {
                "torch_dtype": "auto",  # 使用auto而不是强制float16，更兼容
                "device_map": "auto",
                "low_cpu_mem_usage": True,
            }
            
            if use_flash_attn:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("⚡ Using Flash Attention 2")
            
            print("📦 Loading model weights...")
            _shared_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
            
            print("📦 Loading processor...")
            _shared_processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
            
            if not save_audio:
                print("🔇 Disabling audio output...")
                _shared_model.disable_talker()
            
            print("✅ Model loaded successfully, ready for processing")
            
            load_time = time.time() - start_time
            print(f"⏱️ Total model loading time: {load_time:.2f}s")
    
    return _shared_model, _shared_processor

def calculate_file_hash(file_path: str) -> str:
    """计算文件哈希值用于唯一标识"""
    hasher = hashlib.md5()
    hasher.update(file_path.encode('utf-8'))
    
    # 添加文件大小和修改时间
    try:
        stat = os.stat(file_path)
        hasher.update(str(stat.st_size).encode('utf-8'))
        hasher.update(str(stat.st_mtime).encode('utf-8'))
    except:
        pass
    
    return hasher.hexdigest()

def get_video_duration_fast(video_path: str) -> Optional[float]:
    """快速获取视频时长"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None

def batch_get_video_info(video_paths: List[str], max_workers: int = 16) -> Dict[str, Dict]:
    """
    高速批量获取视频信息 - 采用batch版本策略
    """
    print(f"📊 Fast analyzing {len(video_paths)} videos...")
    
    def get_single_info(video_path):
        try:
            stat = os.stat(video_path)
            duration = get_video_duration_fast(video_path)
            return video_path, {
                'size': stat.st_size,
                'duration': duration,
                'mtime': stat.st_mtime
            }
        except Exception as e:
            return video_path, {'error': str(e)}
    
    results = {}
    # 提高并发数以加速信息获取
    max_workers = min(max_workers, len(video_paths))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_single_info, path): path for path in video_paths}
        
        completed = 0
        for future in as_completed(futures):
            video_path, info = future.result()
            results[video_path] = info
            completed += 1
            if completed % 20 == 0:
                print(f"📊 Analyzed {completed}/{len(video_paths)} videos...")
    
    print(f"✅ Video analysis completed")
    return results

# 移动到模块级别以避免pickle问题
def create_video_segment(args_tuple):
    """创建视频分段的辅助函数"""
    video_path, temp_dir, segment_count, start_time, end_time, actual_duration = args_tuple
    
    segment_filename = f"segment_{segment_count:04d}.mp4"
    segment_path = os.path.join(temp_dir, segment_filename)
    
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-ss', str(start_time), '-t', str(actual_duration),
        '-c:v', 'libx264', '-preset', 'ultrafast',  # 最快编码
        '-c:a', 'aac', '-avoid_negative_ts', 'make_zero',
        segment_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and os.path.exists(segment_path):
            return (segment_path, start_time, end_time)
    except Exception as e:
        print(f"❌ Failed to create segment {segment_count}: {e}")
    
    return None

def split_video_optimized(video_path: str, segment_duration: int = 10, 
                         overlap_duration: int = 2) -> Tuple[List[Tuple[str, float, float]], str]:
    """
    优化的视频分割 - 修复pickle问题
    """
    duration = get_video_duration_fast(video_path)
    if duration is None or duration <= segment_duration:
        return [], ""
    
    temp_dir = tempfile.mkdtemp(prefix="video_segments_opt_")
    segments = []
    
    # 计算所有分割点
    segment_tasks = []
    start_time = 0
    segment_count = 0
    
    while start_time < duration:
        end_time = min(start_time + segment_duration, duration)
        actual_duration = end_time - start_time
        
        if actual_duration < 2:
            break
        
        # 准备参数元组
        task_args = (video_path, temp_dir, segment_count, start_time, end_time, actual_duration)
        segment_tasks.append(task_args)
        
        start_time = end_time - overlap_duration
        if start_time >= duration - overlap_duration:
            break
        segment_count += 1
    
    print(f"🔪 Creating {len(segment_tasks)} segments in parallel...")
    
    # 简化：直接串行处理以避免复杂性
    print(f"🔪 Creating segments sequentially for reliability...")
    for task_args in segment_tasks:
        result = create_video_segment(task_args)
        if result:
            segments.append(result)
    
    segments.sort(key=lambda x: x[1])  # 按开始时间排序
    print(f"✅ Created {len(segments)} segments successfully")
    
    return segments, temp_dir



def process_video_task_optimized(task: VideoTask, 
                                progress_manager: ProgressManager,
                                segment_duration: int = 10,
                                overlap_duration: int = 2,
                                disable_segmentation: bool = False,
                                use_audio: bool = True,
                                worker_id: int = 0) -> bool:
    """
    高度优化的视频任务处理函数 - 采用batch版本的简洁高效策略
    """
    
    try:
        start_time = time.time()
        progress_manager.update_task(task.file_hash, 
                                   status='processing', 
                                   start_time=start_time)
        
        print(f"🔄 Worker {worker_id}: Starting {os.path.basename(task.video_path)}")
        
        # 检查是否需要分割 - 简化判断逻辑
        should_segment = (not disable_segmentation and 
                         task.duration and 
                         task.duration > segment_duration)
        
        # 执行翻译 - 使用统一的优化函数
        if should_segment:
            print(f"🔪 Worker {worker_id}: Segmenting video ({task.duration:.1f}s)")
            translation = translate_segmented_video_optimized(
                task.video_path, segment_duration, overlap_duration, use_audio, worker_id
            )
            segments_count = translation.count('\n\n') + 1 if translation else 0
        else:
            print(f"📹 Worker {worker_id}: Processing directly")
            translation = translate_video_optimized(task.video_path, use_audio, worker_id)
            segments_count = 0
        
        # 保存结果 - 简化错误处理
        if translation and translation.strip():
            os.makedirs(os.path.dirname(task.output_path), exist_ok=True)
            
            with open(task.output_path, "w", encoding="utf-8") as f:
                f.write(translation)
            
            # 更新状态
            end_time = time.time()
            processing_time = end_time - start_time
            
            progress_manager.update_task(task.file_hash,
                                       status='completed',
                                       end_time=end_time,
                                       translation_preview=translation[:200],
                                       segments_count=segments_count)
            
            print(f"✅ Worker {worker_id}: Completed {task.relative_path} in {processing_time:.2f}s")
            return True
        else:
            raise ValueError("Empty translation result")
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Worker {worker_id}: Failed {task.relative_path}: {error_msg}")
        
        progress_manager.update_task(task.file_hash,
                                   status='failed',
                                   error_msg=error_msg,
                                   end_time=time.time())
        return False

def translate_video_optimized(video_path: str, use_audio: bool = True, worker_id: int = 0) -> str:
    """
    优化的视频翻译函数 - 借鉴batch版本的高效策略
    """
    global _gpu_semaphore, _shared_model, _shared_processor
    
    try:
        print(f"🔄 Worker {worker_id}: Processing {os.path.basename(video_path)}")
        
        # 获取共享模型
        model, processor = _shared_model, _shared_processor
        
        # 准备对话
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": "翻译提供的视频中的说话内容到中文。只需要输出翻译内容原文，不要输出任何解释。"}
                ]
            }
        ]

        # 处理输入
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, 
                          return_tensors="pt", padding=True, use_audio_in_video=use_audio)
        inputs = inputs.to(model.device).to(model.dtype)

        # GPU并发控制 - 简化版本
        if _gpu_semaphore:
            _gpu_semaphore.acquire()
        
        try:
            # 生成翻译 - 使用torch.no_grad节省内存
            with torch.no_grad():
                text_ids = model.generate(**inputs, use_audio_in_video=use_audio, 
                                        return_audio=False, max_new_tokens=512)
        finally:
            if _gpu_semaphore:
                _gpu_semaphore.release()
        
        # 解码翻译
        translation = processor.batch_decode(text_ids, skip_special_tokens=True, 
                                           clean_up_tokenization_spaces=False)[0]
        
        print(f"✅ Worker {worker_id}: Completed {os.path.basename(video_path)}")
        return translation.strip()
        
    except Exception as e:
        print(f"❌ Worker {worker_id}: Error translating {video_path}: {str(e)}")
        return ""


def translate_segmented_video_optimized(video_path: str, segment_duration: int, 
                                       overlap_duration: int, use_audio: bool = True, 
                                       worker_id: int = 0) -> str:
    """
    优化的分段视频翻译函数
    """
    segments, temp_dir = split_video_optimized(video_path, segment_duration, overlap_duration)
    
    try:
        if not segments:
            raise ValueError("Failed to create video segments")
        
        translations = []
        
        # 处理每个分段 - 使用统一的翻译函数提高效率
        for i, (segment_path, start_time, end_time) in enumerate(segments):
            print(f"  🎬 Worker {worker_id}: Processing segment {i+1}/{len(segments)}: {start_time:.1f}s - {end_time:.1f}s")
            
            # 使用统一的翻译函数，减少代码重复
            translation = translate_video_optimized(segment_path, use_audio, f"{worker_id}.{i+1}")
            
            if translation and translation.strip():
                translations.append(translation.strip())
                print(f"    ✅ Segment {i+1} completed: {translation[:50]}...")
        
        # 合并翻译结果
        if translations:
            combined_result = "\n\n".join(translations)
            print(f"🎉 Worker {worker_id}: Combined {len(translations)}/{len(segments)} segments")
            return combined_result
        else:
            raise ValueError("No segments were successfully translated")
            
    finally:
        # 清理临时文件
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"⚠️ Failed to cleanup {temp_dir}: {e}")

def get_video_files_recursive(folder_path: str, max_depth: int = 10) -> List[Tuple[str, str]]:
    """递归获取视频文件"""
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    video_extensions = [
        '*.mp4', '*.MP4', '*.avi', '*.AVI', '*.mov', '*.MOV', 
        '*.mkv', '*.MKV', '*.wmv', '*.WMV', '*.flv', '*.FLV',
        '*.webm', '*.WEBM', '*.m4v', '*.M4V', '*.3gp', '*.3GP'
    ]
    
    video_files = []
    
    def _recursive_search(current_path, current_depth):
        if current_depth > max_depth:
            return
        
        for pattern in video_extensions:
            for video_file in current_path.glob(pattern):
                if video_file.is_file():
                    relative_path = video_file.relative_to(folder_path)
                    video_files.append((str(video_file), str(relative_path)))
        
        for subdir in current_path.iterdir():
            if subdir.is_dir():
                _recursive_search(subdir, current_depth + 1)
    
    print(f"🔍 Scanning for video files in: {folder_path}")
    _recursive_search(folder_path, 0)
    
    video_files.sort(key=lambda x: x[1])
    print(f"📁 Found {len(video_files)} video files")
    
    return video_files

def run_optimized_parallel_translation(video_files_info: List[Tuple[str, str]], 
                                     output_folder: str,
                                     model_path: str,
                                     checkpoint_file: str,
                                     preserve_structure: bool = True,
                                     segment_duration: int = 10,
                                     overlap_duration: int = 2,
                                     disable_segmentation: bool = False,
                                     parallel_workers: int = 5,
                                     max_concurrent_gpu: int = 2,
                                     use_flash_attn: bool = False,
                                     save_audio: bool = False,
                                     use_audio: bool = True) -> Dict:
    """
    运行高度优化的并行翻译 - 采用batch版本的高效架构
    """
    global _gpu_semaphore
    
    # 初始化GPU信号量
    _gpu_semaphore = threading.Semaphore(max_concurrent_gpu)
    
    # 预加载共享模型（关键优化）
    get_shared_model(model_path, use_flash_attn, save_audio)
    
    # 初始化进度管理器
    progress_manager = ProgressManager(checkpoint_file)
    progress_manager.load_checkpoint()
    
    # 准备任务列表
    print("📋 Preparing task list...")
    
    # 批量获取视频信息
    video_paths = [video_file for video_file, _ in video_files_info]
    video_info_dict = batch_get_video_info(video_paths)
    
    new_tasks = 0
    for video_file, relative_path in video_files_info:
        file_hash = calculate_file_hash(video_file)
        
        # 检查是否已存在
        if file_hash in progress_manager.tasks:
            existing_task = progress_manager.tasks[file_hash]
            if existing_task.status == 'completed' and os.path.exists(existing_task.output_path):
                continue  # 跳过已完成的任务
        
        # 生成输出路径
        if preserve_structure:
            video_path = Path(relative_path)
            output_dir = Path(output_folder) / video_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{video_path.stem}.txt"
        else:
            flat_name = str(Path(relative_path).with_suffix('')).replace(os.sep, '_').replace('/', '_')
            output_path = Path(output_folder) / f"{flat_name}.txt"
        
        # 获取视频信息
        info = video_info_dict.get(video_file, {})
        
        task = VideoTask(
            video_path=video_file,
            relative_path=relative_path,
            output_path=str(output_path),
            file_hash=file_hash,
            file_size=info.get('size', 0),
            duration=info.get('duration')
        )
        
        progress_manager.add_task(task)
        new_tasks += 1
    
    print(f"📊 Task summary: {new_tasks} new tasks added")
    
    # 获取待处理任务
    pending_tasks = progress_manager.get_pending_tasks()
    if not pending_tasks:
        print("✅ All tasks completed!")
        return progress_manager.get_progress_summary()
    
    print(f"🚀 Starting optimized parallel processing: {len(pending_tasks)} tasks")
    
    # 自动保存检查点的定时器
    def auto_save_checkpoint():
        while True:
            time.sleep(30)  # 每30秒保存一次
            progress_manager.save_checkpoint()
    
    checkpoint_thread = threading.Thread(target=auto_save_checkpoint, daemon=True)
    checkpoint_thread.start()
    
    # 高效并行处理 - 借鉴batch版本策略
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {}
        
        for i, task in enumerate(pending_tasks):
            future = executor.submit(
                process_video_task_optimized,
                task, progress_manager,
                segment_duration, overlap_duration, disable_segmentation, use_audio, i + 1
            )
            futures[future] = task
        
        # 处理完成的任务
        completed_count = 0
        for future in as_completed(futures):
            task = futures[future]
            success = future.result()
            completed_count += 1
            
            # 实时进度显示
            progress = progress_manager.get_progress_summary()
            print(f"📊 Progress: {progress['completed']}/{progress['total']} "
                  f"({progress['progress']:.1f}%) - "
                  f"Failed: {progress['failed']}, "
                  f"Remaining: {progress['remaining']}")
            
            # 定期保存检查点
            if completed_count % 5 == 0:
                progress_manager.save_checkpoint()
    
    # 最终保存
    progress_manager.save_checkpoint()
    
    # 返回最终统计
    final_stats = progress_manager.get_progress_summary()
    print(f"\n🎉 Optimized parallel processing completed!")
    print(f"✅ Success: {final_stats['completed']}")
    print(f"❌ Failed: {final_stats['failed']}")
    print(f"📊 Success rate: {final_stats['completed']/final_stats['total']*100:.1f}%")
    
    return final_stats

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Optimized Qwen2.5-Omni Parallel Video Translation")
    
    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video_path", type=str, help="Single video file path")
    input_group.add_argument("--input_folder", type=str, help="Input folder containing videos")
    
    parser.add_argument("--output_path", type=str, help="Output path or folder")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Omni-7B", help="Model path")
    parser.add_argument("--checkpoint_file", type=str, default="./progress_checkpoint.json", 
                       help="Checkpoint file for resume capability")
    
    # 处理选项
    parser.add_argument("--preserve_structure", action="store_true", help="Preserve directory structure")
    parser.add_argument("--segment_duration", type=int, default=10, help="Segment duration (seconds)")
    parser.add_argument("--overlap_duration", type=int, default=2, help="Overlap duration (seconds)")
    parser.add_argument("--disable_segmentation", action="store_true", help="Disable video segmentation")
    
    # 并行处理选项
    parser.add_argument("--parallel_workers", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--max_concurrent_gpu", type=int, default=2, help="Max concurrent GPU operations")
    
    # 模型选项
    parser.add_argument("--use_flash_attn", action="store_true", help="Use Flash Attention 2")
    parser.add_argument("--save_audio", action="store_true", help="Save audio output")
    parser.add_argument("--use_audio", action="store_true", help="Use audio in video")
    
    # 其他选项
    parser.add_argument("--max_depth", type=int, default=10, help="Max recursion depth")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('optimized_translation.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        if args.video_path:
            # 单文件处理
            if not os.path.exists(args.video_path):
                raise FileNotFoundError(f"Video file not found: {args.video_path}")
            
            output_path = args.output_path or "./evaluation/test_data/optimized_result.txt"
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # 创建单个任务
            file_hash = calculate_file_hash(args.video_path)
            duration = get_video_duration_fast(args.video_path)
            
            task = VideoTask(
                video_path=args.video_path,
                relative_path=os.path.basename(args.video_path),
                output_path=output_path,
                file_hash=file_hash,
                file_size=os.path.getsize(args.video_path),
                duration=duration
            )
            
            # 初始化优化架构
            global _gpu_semaphore, _shared_model, _shared_processor
            _gpu_semaphore = threading.Semaphore(1)
            get_shared_model(args.model_path, args.use_flash_attn, args.save_audio)
            progress_manager = ProgressManager(args.checkpoint_file)
            
            # 处理单个视频
            success = process_video_task_optimized(
                task, progress_manager,
                args.segment_duration, args.overlap_duration, args.disable_segmentation, 
                args.use_audio, 1
            )
            
            if success:
                with open(output_path, 'r', encoding='utf-8') as f:
                    translation = f.read()
                print(f"✅ Translation completed and saved to {output_path}")
                print(f"📝 Translation: {translation[:200]}...")
            else:
                print("❌ Translation failed")
        
        else:
            # 批量处理
            video_files_info = get_video_files_recursive(args.input_folder, args.max_depth)
            
            if not video_files_info:
                print("No video files found.")
                return
            
            output_folder = args.output_path or "./evaluation/test_data/optimized_results"
            os.makedirs(output_folder, exist_ok=True)
            
            # 运行高度优化的并行翻译
            stats = run_optimized_parallel_translation(
                video_files_info, output_folder, args.model_path, args.checkpoint_file,
                args.preserve_structure, args.segment_duration, args.overlap_duration,
                args.disable_segmentation, args.parallel_workers, args.max_concurrent_gpu,
                args.use_flash_attn, args.save_audio, args.use_audio
            )
            
            print(f"🎯 Final Results: {stats}")
    
    except KeyboardInterrupt:
        print("\n⏸️ Processing interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        raise

if __name__ == "__main__":
    main() 