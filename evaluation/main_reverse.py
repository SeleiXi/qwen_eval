import os
import argparse
import torch
import soundfile as sf
from pathlib import Path
import glob
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# if you want to run this file, torch is required to be at least 2.6.0.


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni Video Translation (Reverse Order with Batch Processing)")
    
    # Input options - either single video or folder
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video_path", type=str, help="Path to a single video file")
    input_group.add_argument("--input_folder", type=str, help="Path to folder containing video files for batch processing")
    
    parser.add_argument("--output_path", type=str, help="Path to save the translation text (for single file) or output folder (for batch processing)")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Omni-7B", help="Path to the model")
    parser.add_argument("--source_lang", type=str, default="auto", help="Source language (auto for auto-detection)")
    parser.add_argument("--target_lang", type=str, default="en", help="Target language (en, zh, etc.)")
    parser.add_argument("--use_audio", action="store_true", help="Use audio in video for better translation")
    parser.add_argument("--save_audio", action="store_true", help="Save audio output")
    parser.add_argument("--use_flash_attn", action="store_true", help="Use Flash Attention 2 for faster inference")
    
    # Parallel processing parameters
    parser.add_argument("--parallel_workers", type=int, default=5, help="Number of parallel video processing workers (default: 5)")
    parser.add_argument("--shared_model", action="store_true", help="Use shared model for all workers (saves memory)")
    parser.add_argument("--max_concurrent_gpu", type=int, default=2, help="Maximum concurrent GPU operations (default: 2)")
    parser.add_argument("--serial_mode", action="store_true", help="Use serial processing instead of parallel (original behavior)")
    
    return parser.parse_args()

def get_video_files_reverse(folder_path):
    """
    Get all video files from the specified folder in REVERSE alphabetical order
    
    Args:
        folder_path: Path to the folder containing video files
        
    Returns:
        List of video file paths in reverse order (z-a, 9-0)
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Find all video files (common formats, case insensitive)
    video_extensions = [
        '*.mp4', '*.MP4', '*.Mp4', '*.mP4',
        '*.avi', '*.AVI', '*.Avi', '*.aVi',
        '*.mov', '*.MOV', '*.Mov', '*.mOv',
        '*.mkv', '*.MKV', '*.Mkv', '*.mKv',
        '*.wmv', '*.WMV', '*.Wmv', '*.wMv',
        '*.flv', '*.FLV', '*.Flv', '*.fLv',
        '*.webm', '*.WEBM', '*.Webm', '*.wEbm',
        '*.m4v', '*.M4V', '*.M4v', '*.m4V',
        '*.3gp', '*.3GP', '*.3Gp', '*.3gP'
    ]
    
    video_files = []
    for pattern in video_extensions:
        video_files.extend(glob.glob(str(folder_path / pattern)))
    
    if not video_files:
        print(f"Warning: No video files found in folder: {folder_path}")
        print("Supported formats: mp4, avi, mov, mkv, wmv, flv, webm, m4v, 3gp")
        return []
    
    # 🔄 REVERSE ORDER: Sort in descending order (z-a, 9-0)
    return sorted(video_files, reverse=True)

# Global model management for parallel processing
_shared_model = None
_shared_processor = None
_model_lock = threading.Lock()
_gpu_semaphore = None

def get_shared_model(model_path, use_flash_attn=False, save_audio=False):
    """
    获取共享模型实例（线程安全）
    
    Args:
        model_path: Path to the model
        use_flash_attn: Whether to use Flash Attention
        save_audio: Whether to save audio output
        
    Returns:
        tuple: (model, processor)
    """
    global _shared_model, _shared_processor
    
    with _model_lock:
        if _shared_model is None:
            print(f"🤖 Loading shared model from {model_path}...")
            model_kwargs = {
                "torch_dtype": "auto",
                "device_map": "auto",
            }
            
            if use_flash_attn:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            
            _shared_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
            _shared_processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
            
            if not save_audio:
                _shared_model.disable_talker()
            
            print("✅ Shared model loaded successfully")
    
    return _shared_model, _shared_processor

def translate_video(video_path, model_path, source_lang="auto", target_lang="en", 
                   use_audio=True, save_audio=False, use_flash_attn=False):
    """
    Translate content from a video using Qwen2.5-Omni model
    
    Args:
        video_path: Path to the video file
        model_path: Path to the Qwen2.5-Omni model
        source_lang: Source language (auto for auto-detection)
        target_lang: Target language for translation
        use_audio: Whether to use audio in video
        save_audio: Whether to save audio output
        use_flash_attn: Whether to use Flash Attention 2
        
    Returns:
        Translation text
    """
    # Load model and processor
    print(f"Loading model from {model_path}...")
    model_kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto",
    }
    
    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    
    if not save_audio:
        model.disable_talker()
    
    # Prepare conversation with translation instruction
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

    # Process the input
    print("Processing video...")
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, 
                      return_tensors="pt", padding=True, use_audio_in_video=use_audio)
    inputs = inputs.to(model.device).to(model.dtype)

    # Generate translation
    print("Generating translation...")
    if save_audio:
        print("saving audio")
        text_ids, audio = model.generate(**inputs, use_audio_in_video=use_audio)
    else:
        text_ids = model.generate(**inputs, use_audio_in_video=use_audio, return_audio=False)
    
    # Decode translation
    print("decoding")
    translation = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    # Save audio if requested
    if save_audio:
        print("saving audio")
        audio_output_path = os.path.splitext(video_path)[0] + "_translation.wav"
        sf.write(
            audio_output_path,
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )
        print(f"Audio saved to {audio_output_path}")
    
    return translation

def translate_video_worker(video_path, model_path, source_lang="auto", target_lang="en", 
                          use_audio=True, save_audio=False, use_flash_attn=False, 
                          shared_model=False, worker_id=0):
    """
    工作线程的视频翻译函数
    
    Args:
        video_path: Path to the video file
        model_path: Path to the Qwen2.5-Omni model
        source_lang: Source language (auto for auto-detection)
        target_lang: Target language for translation
        use_audio: Whether to use audio in video
        save_audio: Whether to save audio output
        use_flash_attn: Whether to use Flash Attention 2
        shared_model: Whether to use shared model
        worker_id: Worker ID for logging
        
    Returns:
        Translation text
    """
    global _gpu_semaphore
    
    try:
        print(f"🔄 Worker {worker_id}: Starting translation for {os.path.basename(video_path)}")
        
        # 获取模型
        if shared_model:
            model, processor = get_shared_model(model_path, use_flash_attn, save_audio)
        else:
            # 独立加载模型
            print(f"🤖 Worker {worker_id}: Loading independent model...")
            model_kwargs = {
                "torch_dtype": "auto",
                "device_map": "auto",
            }
            
            if use_flash_attn:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
            processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
            
            if not save_audio:
                model.disable_talker()
        
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

        # 处理输入
        print(f"📊 Worker {worker_id}: Processing video...")
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, 
                          return_tensors="pt", padding=True, use_audio_in_video=use_audio)
        inputs = inputs.to(model.device).to(model.dtype)

        # GPU并发控制
        if _gpu_semaphore:
            _gpu_semaphore.acquire()
        
        try:
            # 生成翻译
            print(f"🤖 Worker {worker_id}: Generating translation...")
            with torch.no_grad():  # 节省内存
                if save_audio:
                    text_ids, audio = model.generate(**inputs, use_audio_in_video=use_audio, max_new_tokens=512)
                    # Save audio for this worker
                    audio_output_path = os.path.splitext(video_path)[0] + f"_worker{worker_id}_translation.wav"
                    sf.write(
                        audio_output_path,
                        audio.reshape(-1).detach().cpu().numpy(),
                        samplerate=24000,
                    )
                    print(f"🎵 Worker {worker_id}: Audio saved to {audio_output_path}")
                else:
                    text_ids = model.generate(**inputs, use_audio_in_video=use_audio, return_audio=False, max_new_tokens=512)
        finally:
            if _gpu_semaphore:
                _gpu_semaphore.release()
        
        # 解码翻译
        print(f"📝 Worker {worker_id}: Decoding...")
        translation = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # 清理独立模型
        if not shared_model:
            del model, processor
            torch.cuda.empty_cache()
        
        print(f"✅ Worker {worker_id}: Completed translation for {os.path.basename(video_path)}")
        return translation.strip()
        
    except Exception as e:
        print(f"❌ Worker {worker_id}: Error translating {video_path}: {str(e)}")
        return ""

def reverse_parallel_batch_translate(video_files, output_folder, model_path, source_lang="auto", target_lang="en",
                                    use_audio=True, save_audio=False, use_flash_attn=False,
                                    parallel_workers=5, shared_model=False, max_concurrent_gpu=2):
    """
    反向顺序并行批量翻译视频文件
    
    Args:
        video_files: List of video file paths (already in reverse order)
        output_folder: Output folder for translations
        model_path: Path to the model
        source_lang: Source language
        target_lang: Target language
        use_audio: Whether to use audio
        save_audio: Whether to save audio
        use_flash_attn: Whether to use Flash Attention
        parallel_workers: Number of parallel workers
        shared_model: Whether to use shared model
        max_concurrent_gpu: Maximum concurrent GPU operations
        
    Returns:
        Dict with processing statistics
    """
    global _gpu_semaphore
    
    # 初始化GPU信号量
    _gpu_semaphore = threading.Semaphore(max_concurrent_gpu)
    
    # 预加载共享模型（如果启用）
    if shared_model:
        get_shared_model(model_path, use_flash_attn, save_audio)
    
    # 准备任务队列
    tasks = []
    for video_file in video_files:
        video_name = Path(video_file).stem
        output_filename = f"{video_name}.txt"
        output_path = os.path.join(output_folder, output_filename)
        
        # 检查输出文件是否已存在
        if os.path.exists(output_path):
            print(f"⏭️ Skipping existing: {os.path.basename(video_file)}")
            continue
        
        tasks.append((video_file, output_path))
    
    if not tasks:
        print("No new videos to process.")
        return {"skipped": len(video_files), "successful": 0, "failed": 0}
    
    print(f"🚀 Starting REVERSE ORDER parallel processing of {len(tasks)} videos with {parallel_workers} workers")
    print(f"📊 Configuration: Shared Model = {shared_model}, Max GPU Concurrent = {max_concurrent_gpu}")
    print(f"🔄 Processing order: REVERSE (z-a, 9-0)")
    
    # 统计计数器
    successful_count = 0
    failed_count = 0
    skipped_count = len(video_files) - len(tasks)
    
    start_time = time.time()
    
    # 使用线程池执行并行翻译
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        # 提交所有任务
        future_to_video = {}
        for i, (video_file, output_path) in enumerate(tasks):
            future = executor.submit(
                translate_video_worker,
                video_file, model_path, source_lang, target_lang,
                use_audio, save_audio, use_flash_attn, shared_model, i+1
            )
            future_to_video[future] = (video_file, output_path)
        
        # 处理完成的任务
        for future in as_completed(future_to_video):
            video_file, output_path = future_to_video[future]
            
            try:
                translation = future.result()
                
                if translation and translation.strip():
                    # 保存翻译结果
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(translation)
                    
                    print(f"✅ Saved: {os.path.basename(output_path)}")
                    print(f"📝 Preview: {translation[:50]}...")
                    successful_count += 1
                else:
                    print(f"⚠️ Empty translation for: {os.path.basename(video_file)}")
                    failed_count += 1
                    
            except Exception as e:
                print(f"❌ Task failed for {os.path.basename(video_file)}: {e}")
                failed_count += 1
    
    elapsed_time = time.time() - start_time
    
    # 输出统计信息
    print(f"\n{'='*80}")
    print(f"🎉 REVERSE ORDER PARALLEL PROCESSING COMPLETED!")
    print(f"{'='*80}")
    print(f"⏱️ Total processing time: {elapsed_time/60:.1f} minutes")
    print(f"📊 Processing Summary:")
    print(f"  📁 Total files found: {len(video_files)}")
    print(f"  🔄 Processing order: REVERSE (z-a, 9-0)")
    print(f"  ⏭️ Files skipped (already exist): {skipped_count}")
    print(f"  ✅ Files successfully processed: {successful_count}")
    print(f"  ❌ Files failed to process: {failed_count}")
    print(f"  🚀 Average speed: {successful_count*60/elapsed_time:.1f} videos/hour")
    print(f"  ⚡ Parallel efficiency: {parallel_workers}x workers")
    print(f"  📂 Results saved in: {output_folder}")
    print(f"{'='*80}")
    
    return {
        "successful": successful_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "total_time": elapsed_time
    }

def main():
    args = parse_args()
    
    if args.video_path:
        # Single video processing
        if not os.path.exists(args.video_path):
            raise FileNotFoundError(f"Video file not found: {args.video_path}")
        
        # Set default output path if not provided
        if not args.output_path:
            args.output_path = "./evaluation/test_data/qwen_result.txt"
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        
        # Translate video
        translation = translate_video(
            args.video_path, 
            args.model_path, 
            args.source_lang, 
            args.target_lang,
            args.use_audio,
            args.save_audio,
            args.use_flash_attn
        )
        
        # Save translation to file
        with open(args.output_path, "w", encoding="utf-8") as f:
            f.write(translation)
        
        print(f"Translation saved to {args.output_path}")
        print("\nTranslation result:")
        print("-" * 50)
        print(translation)
        print("-" * 50)
        
    else:
        # Batch processing with REVERSE ORDER
        print(f"🔄 Starting REVERSE ORDER batch processing for folder: {args.input_folder}")
        
        # 🔄 Use reverse order function
        video_files = get_video_files_reverse(args.input_folder)
        
        if not video_files:
            print("No video files found to process. Exiting.")
            return
            
        print(f"📁 Found {len(video_files)} video files to process in REVERSE order")
        print(f"🔄 Processing order: {[os.path.basename(f) for f in video_files[:5]]}{'...' if len(video_files) > 5 else ''}")
        
        # Set default output folder if not provided
        if not args.output_path:
            args.output_path = "./evaluation/test_data/reverse_batch_results"
        
        # Create output directory
        os.makedirs(args.output_path, exist_ok=True)
        
        if args.serial_mode:
            # 原始串行处理模式
            print("🔗 Using SERIAL processing mode (original behavior)")
            
            # Initialize counters for statistics
            skipped_count = 0
            successful_count = 0
            failed_count = 0
            start_time = time.time()
            
            # Process each video file in REVERSE order
            for i, video_file in enumerate(video_files, 1):
                print(f"\n{'='*60}")
                print(f"🔄 Processing file {i}/{len(video_files)} (REVERSE ORDER): {os.path.basename(video_file)}")
                print(f"{'='*60}")
                
                # Generate output filename based on input filename
                video_name = Path(video_file).stem
                output_filename = f"{video_name}.txt"
                output_path = os.path.join(args.output_path, output_filename)
                
                # Check if output file already exists
                if os.path.exists(output_path):
                    print(f"⏭️  Skipping: Translation file already exists at {output_path}")
                    skipped_count += 1
                    continue
                
                try:
                    translation = translate_video(
                        video_file,
                        args.model_path,
                        args.source_lang,
                        args.target_lang,
                        args.use_audio,
                        args.save_audio,
                        args.use_flash_attn
                    )
                    
                    # Save translation to file
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(translation)
                    
                    print(f"✓ Translation saved to {output_path}")
                    print(f"Translation preview: {translation[:100]}...")
                    successful_count += 1
                    
                except Exception as e:
                    print(f"✗ Error processing {video_file}: {str(e)}")
                    failed_count += 1
                    continue
            
            elapsed_time = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"🔄 REVERSE ORDER SERIAL PROCESSING COMPLETED!")
            print(f"{'='*60}")
            print(f"⏱️ Total processing time: {elapsed_time/60:.1f} minutes")
            print(f"📊 Processing Summary:")
            print(f"  📁 Total files found: {len(video_files)}")
            print(f"  🔄 Processing order: REVERSE (z-a, 9-0)")
            print(f"  🔗 Processing mode: SERIAL")
            print(f"  ⏭️  Files skipped (already exist): {skipped_count}")
            print(f"  ✅ Files successfully processed: {successful_count}")
            print(f"  ❌ Files failed to process: {failed_count}")
            print(f"  📂 Results saved in: {args.output_path}")
            print(f"{'='*60}")
        else:
            # 新的并行处理模式
            print("⚡ Using PARALLEL processing mode (new feature)")
            
            # 执行反向顺序并行批量翻译
            stats = reverse_parallel_batch_translate(
                video_files,
                args.output_path,
                args.model_path,
                args.source_lang,
                args.target_lang,
                args.use_audio,
                args.save_audio,
                args.use_flash_attn,
                args.parallel_workers,
                args.shared_model,
                args.max_concurrent_gpu
            )

if __name__ == "__main__":
    main() 