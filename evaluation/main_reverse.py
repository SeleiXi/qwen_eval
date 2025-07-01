import os
import argparse
import torch
import soundfile as sf
from pathlib import Path
import glob

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# if you want to run this file, torch is required to be at least 2.6.0.


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni Video Translation (Reverse Order)")
    
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
        
        # Initialize counters for statistics
        skipped_count = 0
        successful_count = 0
        failed_count = 0
        
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
        
        print(f"\n{'='*60}")
        print(f"🔄 REVERSE ORDER BATCH PROCESSING COMPLETED!")
        print(f"{'='*60}")
        print(f"📊 Processing Summary:")
        print(f"  📁 Total files found: {len(video_files)}")
        print(f"  🔄 Processing order: REVERSE (z-a, 9-0)")
        print(f"  ⏭️  Files skipped (already exist): {skipped_count}")
        print(f"  ✅ Files successfully processed: {successful_count}")
        print(f"  ❌ Files failed to process: {failed_count}")
        print(f"  📂 Results saved in: {args.output_path}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main() 