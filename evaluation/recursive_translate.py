import os
import argparse
import torch
import soundfile as sf
from pathlib import Path
import glob
import subprocess
import tempfile
import shutil

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# if you want to run this file, torch is required to be at least 2.6.0.


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni Recursive Video Translation with Auto-Segmentation")
    
    # Input options - either single video or folder (recursive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video_path", type=str, help="Path to a single video file")
    input_group.add_argument("--input_folder", type=str, help="Path to folder containing video files for recursive batch processing")
    
    parser.add_argument("--output_path", type=str, help="Path to save the translation text (for single file) or output folder (for batch processing)")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Omni-7B", help="Path to the model")
    parser.add_argument("--source_lang", type=str, default="auto", help="Source language (auto for auto-detection)")
    parser.add_argument("--target_lang", type=str, default="en", help="Target language (en, zh, etc.)")
    parser.add_argument("--use_audio", action="store_true", help="Use audio in video for better translation")
    parser.add_argument("--save_audio", action="store_true", help="Save audio output")
    parser.add_argument("--use_flash_attn", action="store_true", help="Use Flash Attention 2 for faster inference")
    parser.add_argument("--preserve_structure", action="store_true", help="Preserve directory structure in output folder")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum recursion depth for subdirectories (default: 10)")
    
    # Video segmentation parameters
    parser.add_argument("--segment_duration", type=int, default=30, help="Duration of each segment in seconds (default: 30s)")
    parser.add_argument("--disable_segmentation", action="store_true", help="Disable automatic video segmentation")
    parser.add_argument("--overlap_duration", type=int, default=2, help="Overlap between segments in seconds (default: 2s)")
    
    return parser.parse_args()

def get_video_files_recursive(folder_path, max_depth=10):
    """
    Recursively get all video files from the specified folder and its subdirectories
    
    Args:
        folder_path: Path to the folder containing video files
        max_depth: Maximum recursion depth to prevent infinite loops
        
    Returns:
        List of tuples: (video_file_path, relative_path_from_root)
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Supported video extensions
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
    
    def _recursive_search(current_path, current_depth):
        if current_depth > max_depth:
            print(f"Warning: Maximum depth ({max_depth}) reached at {current_path}")
            return
        
        # Search for video files in current directory
        for pattern in video_extensions:
            for video_file in current_path.glob(pattern):
                if video_file.is_file():
                    # Calculate relative path from the root input folder
                    relative_path = video_file.relative_to(folder_path)
                    video_files.append((str(video_file), str(relative_path)))
        
        # Recursively search subdirectories
        for subdir in current_path.iterdir():
            if subdir.is_dir():
                _recursive_search(subdir, current_depth + 1)
    
    print(f"🔍 Recursively searching for video files in: {folder_path}")
    _recursive_search(folder_path, 0)
    
    if not video_files:
        print(f"Warning: No video files found in folder: {folder_path}")
        print("Supported formats: mp4, avi, mov, mkv, wmv, flv, webm, m4v, 3gp")
        return []
    
    # Sort by relative path for consistent processing order
    video_files.sort(key=lambda x: x[1])
    
    print(f"📁 Found {len(video_files)} video files across all subdirectories")
    
    # Display directory structure
    dirs = set()
    for _, rel_path in video_files:
        dir_path = str(Path(rel_path).parent)
        if dir_path != '.':
            dirs.add(dir_path)
    
    if dirs:
        print("📂 Directories containing videos:")
        for dir_path in sorted(dirs):
            count = sum(1 for _, rel_path in video_files if str(Path(rel_path).parent) == dir_path)
            print(f"  {dir_path}: {count} files")
    
    return video_files

def get_video_duration(video_path):
    """
    Get video duration using ffprobe
    
    Args:
        video_path: Path to the video file
        
    Returns:
        float: Duration in seconds, or None if failed
    """
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            return duration
    except Exception as e:
        print(f"⚠️ Warning: Could not get duration for {os.path.basename(video_path)}: {e}")
    return None

def split_video_into_segments(video_path, segment_duration=30, overlap_duration=2):
    """
    Split a video into smaller segments using ffmpeg
    
    Args:
        video_path: Path to the input video
        segment_duration: Duration of each segment in seconds
        overlap_duration: Overlap between segments in seconds
        
    Returns:
        tuple: (segments_list, temp_directory)
    """
    duration = get_video_duration(video_path)
    if duration is None:
        raise ValueError(f"Could not determine video duration for {video_path}")
    
    # Create temporary directory for segments
    temp_dir = tempfile.mkdtemp(prefix="video_segments_")
    segments = []
    
    print(f"📹 Video duration: {duration:.1f}s, splitting into {segment_duration}s segments...")
    
    segment_count = 0
    start_time = 0
    
    while start_time < duration:
        # Calculate end time for this segment
        end_time = min(start_time + segment_duration, duration)
        actual_duration = end_time - start_time
        
        # Skip very short segments (less than 2 seconds)
        if actual_duration < 2:
            break
        
        # Generate segment filename
        segment_filename = f"segment_{segment_count:04d}.mp4"
        segment_path = os.path.join(temp_dir, segment_filename)
        
        # Use ffmpeg to extract segment
        cmd = [
            'ffmpeg', '-i', video_path,
            '-ss', str(start_time),
            '-t', str(actual_duration),
            '-c:v', 'libx264',  # Re-encode to ensure compatibility
            '-c:a', 'aac',
            '-avoid_negative_ts', 'make_zero',
            '-y',  # Overwrite output file
            segment_path
        ]
        
        try:
            print(f"  📼 Creating segment {segment_count + 1}: {start_time:.1f}s - {end_time:.1f}s")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and os.path.exists(segment_path):
                segments.append((segment_path, start_time, end_time))
                print(f"  ✅ Segment {segment_count + 1} created successfully")
            else:
                print(f"  ❌ Failed to create segment {segment_count + 1}: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"  ⏰ Timeout creating segment {segment_count + 1}")
        except Exception as e:
            print(f"  ❌ Error creating segment {segment_count + 1}: {e}")
        
        # Move to next segment with overlap consideration
        start_time = end_time - overlap_duration
        if start_time >= duration - overlap_duration:
            break
        
        segment_count += 1
    
    print(f"📦 Created {len(segments)} segments in {temp_dir}")
    return segments, temp_dir

def translate_video_segment(video_path, model, processor, source_lang="auto", target_lang="en", 
                           use_audio=True, save_audio=False, segment_info=None):
    """
    Translate a single video segment
    
    Args:
        video_path: Path to the video segment
        model: Pre-loaded model
        processor: Pre-loaded processor
        source_lang: Source language
        target_lang: Target language
        use_audio: Whether to use audio
        save_audio: Whether to save audio
        segment_info: Tuple of (start_time, end_time) for this segment
        
    Returns:
        Translation text
    """
    if segment_info:
        start_time, end_time = segment_info
        print(f"  🔤 Translating segment: {start_time:.1f}s - {end_time:.1f}s")
    
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
    print("    📊 Processing video segment...")
    try:
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, 
                          return_tensors="pt", padding=True, use_audio_in_video=use_audio)
        inputs = inputs.to(model.device).to(model.dtype)

        # Generate translation
        print("    🤖 Generating translation...")
        if save_audio:
            text_ids, audio = model.generate(**inputs, use_audio_in_video=use_audio)
        else:
            text_ids = model.generate(**inputs, use_audio_in_video=use_audio, return_audio=False)
        
        # Decode translation
        print("    📝 Decoding...")
        translation = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        return translation.strip()
    
    except Exception as e:
        print(f"    ❌ Error translating segment: {e}")
        return ""

def translate_video_with_segmentation(video_path, model_path, source_lang="auto", target_lang="en", 
                                    use_audio=True, save_audio=False, use_flash_attn=False,
                                    segment_duration=30, overlap_duration=2, disable_segmentation=False):
    """
    Translate a video with automatic segmentation if needed
    
    Args:
        video_path: Path to the video file
        model_path: Path to the model
        source_lang: Source language
        target_lang: Target language
        use_audio: Whether to use audio
        save_audio: Whether to save audio
        use_flash_attn: Whether to use Flash Attention
        segment_duration: Duration of each segment
        overlap_duration: Overlap between segments
        disable_segmentation: Whether to disable automatic segmentation
        
    Returns:
        Translation text
    """
    # Check video duration
    duration = get_video_duration(video_path)
    if duration is None:
        print("⚠️ Could not determine video duration, processing as single video")
        disable_segmentation = True
    
    # Load model and processor once
    print(f"🤖 Loading model from {model_path}...")
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
    
    # Decide whether to segment the video
    should_segment = not disable_segmentation and duration and duration > segment_duration
    
    if not should_segment:
        print(f"✅ Video duration ({duration:.1f}s) <= segment duration ({segment_duration}s), processing directly")
        return translate_video_segment(video_path, model, processor, source_lang, target_lang, 
                                     use_audio, save_audio)
    
    print(f"📹 Video duration ({duration:.1f}s) > segment duration ({segment_duration}s), segmenting...")
    
    # Split video into segments
    segments, temp_dir = split_video_into_segments(video_path, segment_duration, overlap_duration)
    
    if not segments:
        raise ValueError("Failed to create video segments")
    
    all_translations = []
    
    try:
        # Process each segment
        for i, (segment_path, start_time, end_time) in enumerate(segments):
            print(f"\n🎬 Processing segment {i+1}/{len(segments)}")
            try:
                translation = translate_video_segment(
                    segment_path, model, processor, source_lang, target_lang, 
                    use_audio, False, (start_time, end_time)  # Don't save audio for segments
                )
                
                if translation and translation.strip():
                    all_translations.append(translation.strip())
                    print(f"  ✅ Segment {i+1} completed: {translation[:50]}...")
                else:
                    print(f"  ⚠️ Segment {i+1} produced empty translation")
                    
            except Exception as e:
                print(f"  ❌ Segment {i+1} failed: {e}")
                continue
        
        # Combine all translations
        if all_translations:
            # Use double newline to separate segments
            final_translation = "\n\n".join(all_translations)
            print(f"🎉 Combined translation from {len(all_translations)}/{len(segments)} segments")
            return final_translation
        else:
            raise ValueError("No segments were successfully translated")
            
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"🧹 Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"⚠️ Failed to clean up temporary directory: {e}")

def generate_output_path(relative_path, output_folder, preserve_structure):
    """
    Generate output file path based on input relative path and options
    
    Args:
        relative_path: Relative path of the video file from input root
        output_folder: Base output folder
        preserve_structure: Whether to preserve directory structure
        
    Returns:
        Output file path for the translation
    """
    video_path = Path(relative_path)
    video_name = video_path.stem
    
    if preserve_structure:
        # Preserve directory structure
        output_dir = Path(output_folder) / video_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_name}.txt"
    else:
        # Flat structure with path-based naming to avoid conflicts
        # Replace directory separators with underscores
        flat_name = str(video_path.with_suffix('')).replace(os.sep, '_').replace('/', '_')
        output_path = Path(output_folder) / f"{flat_name}.txt"
    
    return str(output_path)

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
        
        # Check video duration and display info
        duration = get_video_duration(args.video_path)
        if duration:
            print(f"📹 Video duration: {duration:.1f}s ({duration/60:.1f} minutes)")
            if duration > args.segment_duration and not args.disable_segmentation:
                print(f"🔪 Will segment into {args.segment_duration}s chunks")
        
        # Translate video
        translation = translate_video_with_segmentation(
            args.video_path, 
            args.model_path, 
            args.source_lang, 
            args.target_lang,
            args.use_audio,
            args.save_audio,
            args.use_flash_attn,
            args.segment_duration,
            args.overlap_duration,
            args.disable_segmentation
        )
        
        # Save translation to file
        with open(args.output_path, "w", encoding="utf-8") as f:
            f.write(translation)
        
        print(f"✅ Translation saved to {args.output_path}")
        print("\n📝 Translation result:")
        print("-" * 50)
        print(translation)
        print("-" * 50)
        
    else:
        # Recursive batch processing
        print(f"🚀 Starting recursive batch processing for folder: {args.input_folder}")
        video_files_info = get_video_files_recursive(args.input_folder, args.max_depth)
        
        if not video_files_info:
            print("No video files found to process. Exiting.")
            return
        
        # Set default output folder if not provided
        if not args.output_path:
            args.output_path = "./evaluation/test_data/recursive_results"
        
        # Create output directory
        os.makedirs(args.output_path, exist_ok=True)
        
        # Initialize counters for statistics
        skipped_count = 0
        successful_count = 0
        failed_count = 0
        segmented_count = 0
        
        # Process each video file
        for i, (video_file, relative_path) in enumerate(video_files_info, 1):
            print(f"\n{'='*80}")
            print(f"Processing file {i}/{len(video_files_info)}")
            print(f"📁 Path: {relative_path}")
            print(f"🎬 File: {os.path.basename(video_file)}")
            
            # Display video duration info
            duration = get_video_duration(video_file)
            if duration:
                print(f"⏱️ Duration: {duration:.1f}s ({duration/60:.1f} min)")
                if duration > args.segment_duration and not args.disable_segmentation:
                    print(f"🔪 Will be segmented into {args.segment_duration}s chunks")
                    segmented_count += 1
            
            print(f"{'='*80}")
            
            # Generate output path
            output_path = generate_output_path(relative_path, args.output_path, args.preserve_structure)
            
            # Check if output file already exists
            if os.path.exists(output_path):
                print(f"⏭️  Skipping: Translation file already exists at {output_path}")
                skipped_count += 1
                continue
            
            try:
                translation = translate_video_with_segmentation(
                    video_file,
                    args.model_path,
                    args.source_lang,
                    args.target_lang,
                    args.use_audio,
                    args.save_audio,
                    args.use_flash_attn,
                    args.segment_duration,
                    args.overlap_duration,
                    args.disable_segmentation
                )
                
                # Create output directory if needed (for preserve_structure mode)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save translation to file
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(translation)
                
                print(f"✅ Translation saved to {output_path}")
                print(f"📝 Preview: {translation[:100]}...")
                successful_count += 1
                
            except Exception as e:
                print(f"❌ Error processing {video_file}: {str(e)}")
                failed_count += 1
                continue
        
        print(f"\n{'='*80}")
        print(f"🎉 RECURSIVE BATCH PROCESSING COMPLETED!")
        print(f"{'='*80}")
        print(f"📊 Processing Summary:")
        print(f"  📁 Total files found: {len(video_files_info)}")
        print(f"  ⏭️  Files skipped (already exist): {skipped_count}")
        print(f"  ✅ Files successfully processed: {successful_count}")
        print(f"  🔪 Files requiring segmentation: {segmented_count}")
        print(f"  ❌ Files failed to process: {failed_count}")
        print(f"  📂 Results saved in: {args.output_path}")
        print(f"  🏗️  Structure preserved: {'Yes' if args.preserve_structure else 'No (flat)'}")
        print(f"{'='*80}")

if __name__ == "__main__":
    main() 