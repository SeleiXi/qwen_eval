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
from concurrent.futures import ThreadPoolExecutor

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni Optimized Video Translation")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video_path", type=str, help="Path to a single video file")
    input_group.add_argument("--input_folder", type=str, help="Path to folder containing video files")
    
    parser.add_argument("--output_path", type=str, help="Output path")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Omni-7B", help="Model path")
    parser.add_argument("--source_lang", type=str, default="auto", help="Source language")
    parser.add_argument("--target_lang", type=str, default="en", help="Target language")
    parser.add_argument("--use_audio", action="store_true", help="Use audio")
    parser.add_argument("--save_audio", action="store_true", help="Save audio")
    parser.add_argument("--use_flash_attn", action="store_true", help="Use Flash Attention")
    parser.add_argument("--preserve_structure", action="store_true", help="Preserve directory structure")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum recursion depth")
    
    # 性能优化选项
    parser.add_argument("--max_duration", type=int, default=10, help="Maximum video duration to process (seconds)")
    parser.add_argument("--skip_validation", action="store_true", help="Skip video file validation for speed")
    parser.add_argument("--batch_model_loading", action="store_true", help="Load model once for batch processing")
    parser.add_argument("--parallel_validation", action="store_true", help="Parallel video validation")
    
    return parser.parse_args()

def get_video_info_batch(video_files, skip_validation=False, parallel=False):
    """
    批量获取视频信息，减少ffprobe调用次数
    
    Args:
        video_files: 视频文件列表
        skip_validation: 跳过验证以提升速度
        parallel: 并行处理
        
    Returns:
        Dict of video_path: {duration, valid}
    """
    print(f"🔍 Analyzing {len(video_files)} video files...")
    start_time = time.time()
    
    def analyze_single_video(video_path):
        """分析单个视频文件"""
        try:
            # 一次ffprobe调用获取所有需要的信息
            cmd = [
                'ffprobe', '-v', 'error', 
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height:format=duration',
                '-of', 'csv=p=0', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    # width,height
                    dimensions = lines[0].split(',')
                    # duration
                    duration = float(lines[1]) if lines[1] else 0
                    
                    valid = not skip_validation and len(dimensions) == 2 and duration > 0
                    return {
                        'duration': duration,
                        'valid': valid or skip_validation,
                        'width': int(dimensions[0]) if dimensions[0] else 0,
                        'height': int(dimensions[1]) if dimensions[1] else 0
                    }
        except Exception as e:
            print(f"  ⚠️ Analysis failed for {os.path.basename(video_path)}: {e}")
        
        return {'duration': 0, 'valid': False, 'width': 0, 'height': 0}
    
    # 根据是否并行处理选择执行方式
    if parallel and len(video_files) > 5:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(analyze_single_video, video_files))
    else:
        results = [analyze_single_video(video) for video in video_files]
    
    # 构建结果字典
    video_info = {}
    valid_count = 0
    total_duration = 0
    
    for video_path, info in zip(video_files, results):
        video_info[video_path] = info
        if info['valid']:
            valid_count += 1
            total_duration += info['duration']
    
    elapsed_time = time.time() - start_time
    print(f"📊 Analysis completed in {elapsed_time:.1f}s:")
    print(f"  ✅ Valid videos: {valid_count}/{len(video_files)}")
    print(f"  ⏱️  Total duration: {total_duration/60:.1f} minutes")
    print(f"  🚀 Speed: {len(video_files)/elapsed_time:.1f} files/second")
    
    return video_info

def get_video_files_optimized(folder_path, max_depth=10, skip_validation=False, parallel=False):
    """
    优化的递归视频文件获取
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # 支持的视频扩展名（预编译模式提升匹配速度）
    video_extensions = {
        '.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV', '.mkv', '.MKV',
        '.wmv', '.WMV', '.flv', '.FLV', '.webm', '.WEBM', '.m4v', '.M4V',
        '.3gp', '.3GP'
    }
    
    print(f"🔍 Scanning directory: {folder_path}")
    start_time = time.time()
    
    # 使用rglob进行快速递归搜索
    all_files = []
    for file_path in folder_path.rglob("*"):
        if (file_path.is_file() and 
            file_path.suffix in video_extensions and
            len(file_path.parts) - len(folder_path.parts) <= max_depth):
            all_files.append(str(file_path))
    
    scan_time = time.time() - start_time
    print(f"📁 Found {len(all_files)} potential video files in {scan_time:.1f}s")
    
    if not all_files:
        print("Warning: No video files found")
        return []
    
    # 批量分析视频信息
    video_info = get_video_info_batch(all_files, skip_validation, parallel)
    
    # 过滤有效视频并生成相对路径
    valid_videos = []
    for video_path, info in video_info.items():
        if info['valid']:
            relative_path = Path(video_path).relative_to(folder_path)
            valid_videos.append((video_path, str(relative_path), info))
    
    # 按路径排序
    valid_videos.sort(key=lambda x: x[1])
    
    print(f"✅ Final result: {len(valid_videos)} valid videos")
    return valid_videos

def translate_video_optimized(video_path, model, processor, source_lang="auto", target_lang="en", 
                            use_audio=True, save_audio=False, max_duration=10):
    """
    优化的视频翻译函数
    """
    print(f"🎬 Translating: {os.path.basename(video_path)}")
    
    # 准备对话
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": f"翻译提供的视频中的说话内容到中文。只需要输出翻译内容原文，不要输出任何解释。"}
            ],
        },
    ]

    try:
        # 处理输入
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, 
                          return_tensors="pt", padding=True, use_audio_in_video=use_audio)
        inputs = inputs.to(model.device).to(model.dtype)

        # 生成翻译
        with torch.no_grad():  # 禁用梯度计算节省内存
            if save_audio:
                text_ids, audio = model.generate(**inputs, use_audio_in_video=use_audio, max_new_tokens=512)
            else:
                text_ids = model.generate(**inputs, use_audio_in_video=use_audio, return_audio=False, max_new_tokens=512)
        
        # 解码
        translation = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return translation.strip()
        
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        return ""

def main():
    args = parse_args()
    
    if args.video_path:
        # 单文件处理模式
        if not os.path.exists(args.video_path):
            raise FileNotFoundError(f"Video file not found: {args.video_path}")
        
        if not args.output_path:
            args.output_path = "./evaluation/test_data/qwen_result.txt"
        
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        
        # 检查视频时长
        if not args.skip_validation:
            video_info = get_video_info_batch([args.video_path], args.skip_validation, False)
            duration = video_info[args.video_path]['duration']
            
            if duration > args.max_duration:
                print(f"⚠️ Video duration ({duration:.1f}s) exceeds maximum ({args.max_duration}s)")
                print("Consider using --max_duration option or video segmentation")
                return
            
            print(f"📹 Video duration: {duration:.1f}s - within limits")
        
        # 加载模型
        print(f"🤖 Loading model from {args.model_path}...")
        model_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
        if args.use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
        processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
        
        if not args.save_audio:
            model.disable_talker()
        
        # 翻译视频
        translation = translate_video_optimized(
            args.video_path, model, processor, 
            args.source_lang, args.target_lang,
            args.use_audio, args.save_audio, args.max_duration
        )
        
        # 保存结果
        with open(args.output_path, "w", encoding="utf-8") as f:
            f.write(translation)
        
        print(f"✅ Translation saved to {args.output_path}")
        print(f"📝 Result: {translation[:100]}...")
        
    else:
        # 批量处理模式
        print(f"🚀 Starting optimized batch processing: {args.input_folder}")
        
        # 优化的文件扫描
        video_files_info = get_video_files_optimized(
            args.input_folder, args.max_depth, 
            args.skip_validation, args.parallel_validation
        )
        
        if not video_files_info:
            print("No valid video files found. Exiting.")
            return
        
        # 设置输出目录
        if not args.output_path:
            args.output_path = "./evaluation/test_data/optimized_results"
        os.makedirs(args.output_path, exist_ok=True)
        
        # 过滤超长视频
        valid_videos = []
        skipped_long = 0
        
        for video_file, relative_path, info in video_files_info:
            if info['duration'] <= args.max_duration:
                valid_videos.append((video_file, relative_path, info))
            else:
                skipped_long += 1
                print(f"⏭️ Skipping long video ({info['duration']:.1f}s): {relative_path}")
        
        print(f"📊 Processing {len(valid_videos)} videos (skipped {skipped_long} long videos)")
        
        # 批量模型加载（可选）
        model = None
        processor = None
        
        if args.batch_model_loading:
            print(f"🤖 Loading model once for batch processing...")
            model_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
            if args.use_flash_attn:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
            processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
            
            if not args.save_audio:
                model.disable_talker()
        
        # 处理统计
        successful_count = 0
        failed_count = 0
        skipped_count = 0
        total_start = time.time()
        
        # 处理每个视频
        for i, (video_file, relative_path, info) in enumerate(valid_videos, 1):
            print(f"\n{'='*60}")
            print(f"Processing {i}/{len(valid_videos)}: {os.path.basename(video_file)}")
            print(f"Duration: {info['duration']:.1f}s | Size: {info['width']}x{info['height']}")
            print(f"{'='*60}")
            
            # 生成输出路径
            if args.preserve_structure:
                output_dir = Path(args.output_path) / Path(relative_path).parent
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{Path(relative_path).stem}.txt"
            else:
                flat_name = str(Path(relative_path).with_suffix('')).replace(os.sep, '_')
                output_path = Path(args.output_path) / f"{flat_name}.txt"
            
            # 检查是否已存在
            if os.path.exists(output_path):
                print(f"⏭️ Skipping: Translation already exists")
                skipped_count += 1
                continue
            
            start_time = time.time()
            
            try:
                # 如果没有预加载模型，临时加载
                if not args.batch_model_loading:
                    print(f"🤖 Loading model...")
                    model_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
                    if args.use_flash_attn:
                        model_kwargs["attn_implementation"] = "flash_attention_2"
                    
                    current_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
                    current_processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
                    
                    if not args.save_audio:
                        current_model.disable_talker()
                else:
                    current_model = model
                    current_processor = processor
                
                # 翻译视频
                translation = translate_video_optimized(
                    video_file, current_model, current_processor,
                    args.source_lang, args.target_lang,
                    args.use_audio, args.save_audio, args.max_duration
                )
                
                # 保存结果
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(translation)
                
                elapsed = time.time() - start_time
                print(f"✅ Completed in {elapsed:.1f}s: {translation[:50]}...")
                successful_count += 1
                
                # 如果是临时加载的模型，清理内存
                if not args.batch_model_loading:
                    del current_model, current_processor
                    torch.cuda.empty_cache()
                
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"❌ Failed after {elapsed:.1f}s: {e}")
                failed_count += 1
        
        # 最终统计
        total_elapsed = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"🎉 OPTIMIZED BATCH PROCESSING COMPLETED!")
        print(f"{'='*60}")
        print(f"⏱️ Total time: {total_elapsed/60:.1f} minutes")
        print(f"📊 Processing Summary:")
        print(f"  📁 Videos found: {len(video_files_info)}")
        print(f"  ⏭️ Skipped (long): {skipped_long}")
        print(f"  ⏭️ Skipped (exists): {skipped_count}")
        print(f"  ✅ Successfully processed: {successful_count}")
        print(f"  ❌ Failed: {failed_count}")
        print(f"  🚀 Average speed: {successful_count*60/total_elapsed:.1f} videos/hour")
        print(f"{'='*60}")

if __name__ == "__main__":
    main() 