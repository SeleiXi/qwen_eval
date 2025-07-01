#!/usr/bin/env python3
"""
检查点管理工具
用于查看、清理和恢复 optimized_parallel_translate.py 的检查点状态

Usage:
    python checkpoint_manager.py status ./progress_checkpoint.json
    python checkpoint_manager.py clean ./progress_checkpoint.json
    python checkpoint_manager.py reset ./progress_checkpoint.json
    python checkpoint_manager.py export ./progress_checkpoint.json ./report.txt
"""

import json
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def load_checkpoint(checkpoint_file: str) -> Dict:
    """加载检查点文件"""
    if not os.path.exists(checkpoint_file):
        return None
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载检查点失败: {e}")
        return None


def show_status(checkpoint_file: str):
    """显示检查点状态"""
    data = load_checkpoint(checkpoint_file)
    
    if not data:
        print(f"📂 检查点文件不存在或损坏: {checkpoint_file}")
        return
    
    tasks = data.get('tasks', [])
    stats = data.get('stats', {})
    timestamp = data.get('timestamp', 'Unknown')
    version = data.get('version', '1.0')
    
    print(f"📊 检查点状态报告")
    print(f"{'='*50}")
    print(f"📁 文件: {checkpoint_file}")
    print(f"📅 时间: {timestamp}")
    print(f"🔖 版本: {version}")
    print()
    
    # 统计任务状态
    status_count = {}
    total_size = 0
    total_duration = 0
    segmented_count = 0
    
    for task in tasks:
        status = task.get('status', 'unknown')
        status_count[status] = status_count.get(status, 0) + 1
        total_size += task.get('file_size', 0)
        
        if task.get('duration'):
            total_duration += task['duration']
        
        if task.get('segments_count', 0) > 0:
            segmented_count += 1
    
    print(f"📈 任务统计")
    print(f"  总任务数: {len(tasks)}")
    for status, count in status_count.items():
        emoji = {'completed': '✅', 'failed': '❌', 'processing': '🔄', 'pending': '⏳'}.get(status, '❓')
        print(f"  {emoji} {status}: {count} ({count/len(tasks)*100:.1f}%)")
    
    print()
    print(f"📊 数据统计")
    print(f"  总文件大小: {total_size / 1024**3:.2f} GB")
    print(f"  总视频时长: {total_duration / 3600:.2f} 小时")
    print(f"  需要分段的视频: {segmented_count}")
    
    # 显示失败的任务
    failed_tasks = [task for task in tasks if task.get('status') == 'failed']
    if failed_tasks:
        print()
        print(f"❌ 失败任务 ({len(failed_tasks)}个)")
        for i, task in enumerate(failed_tasks[:5]):  # 只显示前5个
            print(f"  {i+1}. {task.get('relative_path', 'Unknown')}")
            if task.get('error_msg'):
                print(f"     错误: {task['error_msg'][:100]}...")
        
        if len(failed_tasks) > 5:
            print(f"  ... 还有 {len(failed_tasks) - 5} 个失败任务")
    
    # 性能统计
    completed_tasks = [task for task in tasks if task.get('status') == 'completed' and task.get('start_time') and task.get('end_time')]
    if completed_tasks:
        processing_times = [task['end_time'] - task['start_time'] for task in completed_tasks]
        avg_time = sum(processing_times) / len(processing_times)
        
        print()
        print(f"⚡ 性能统计")
        print(f"  平均处理时间: {avg_time:.2f}s/文件")
        print(f"  预计完成剩余: {avg_time * status_count.get('pending', 0) / 60:.1f} 分钟")


def clean_checkpoint(checkpoint_file: str):
    """清理检查点 - 重置失败的任务为待处理"""
    data = load_checkpoint(checkpoint_file)
    
    if not data:
        print(f"❌ 无法加载检查点文件")
        return
    
    tasks = data.get('tasks', [])
    failed_count = 0
    
    for task in tasks:
        if task.get('status') == 'failed':
            task['status'] = 'pending'
            task['error_msg'] = ''
            task['start_time'] = None
            task['end_time'] = None
            failed_count += 1
    
    # 更新统计信息
    if 'stats' in data:
        data['stats']['failed'] = 0
    
    # 保存更新后的检查点
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 清理完成: 重置了 {failed_count} 个失败任务为待处理状态")
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")


def reset_checkpoint(checkpoint_file: str):
    """重置检查点 - 将所有任务重置为待处理"""
    data = load_checkpoint(checkpoint_file)
    
    if not data:
        print(f"❌ 无法加载检查点文件")
        return
    
    confirm = input("⚠️ 这将重置所有任务为待处理状态，确认吗? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ 操作已取消")
        return
    
    tasks = data.get('tasks', [])
    reset_count = 0
    
    for task in tasks:
        if task.get('status') in ['completed', 'failed', 'processing']:
            task['status'] = 'pending'
            task['error_msg'] = ''
            task['start_time'] = None
            task['end_time'] = None
            task['translation_preview'] = ''
            task['segments_count'] = 0
            reset_count += 1
    
    # 重置统计信息
    if 'stats' in data:
        data['stats'] = {
            'total_files': len(tasks),
            'completed': 0,
            'failed': 0,
            'skipped': 0,
            'segmented': 0,
            'total_duration': 0.0,
            'processing_time': 0.0
        }
    
    # 保存更新后的检查点
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 重置完成: 重置了 {reset_count} 个任务")
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")


def export_report(checkpoint_file: str, output_file: str):
    """导出详细报告"""
    data = load_checkpoint(checkpoint_file)
    
    if not data:
        print(f"❌ 无法加载检查点文件")
        return
    
    tasks = data.get('tasks', [])
    stats = data.get('stats', {})
    timestamp = data.get('timestamp', 'Unknown')
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"检查点详细报告\n")
            f.write(f"{'='*50}\n")
            f.write(f"生成时间: {datetime.now().isoformat()}\n")
            f.write(f"检查点文件: {checkpoint_file}\n")
            f.write(f"检查点时间: {timestamp}\n\n")
            
            # 统计信息
            status_count = {}
            for task in tasks:
                status = task.get('status', 'unknown')
                status_count[status] = status_count.get(status, 0) + 1
            
            f.write(f"任务统计:\n")
            f.write(f"  总数: {len(tasks)}\n")
            for status, count in status_count.items():
                f.write(f"  {status}: {count} ({count/len(tasks)*100:.1f}%)\n")
            f.write("\n")
            
            # 详细任务列表
            f.write(f"详细任务列表:\n")
            f.write(f"{'序号':<6} {'状态':<12} {'相对路径':<50} {'时长':<8} {'分段':<6} {'错误信息'}\n")
            f.write(f"{'-'*120}\n")
            
            for i, task in enumerate(tasks, 1):
                status = task.get('status', 'unknown')
                path = task.get('relative_path', 'Unknown')[:48]
                duration = f"{task.get('duration', 0):.1f}s" if task.get('duration') else 'N/A'
                segments = str(task.get('segments_count', 0))
                error = task.get('error_msg', '')[:30] if task.get('error_msg') else ''
                
                f.write(f"{i:<6} {status:<12} {path:<50} {duration:<8} {segments:<6} {error}\n")
        
        print(f"✅ 报告已导出到: {output_file}")
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")


def list_checkpoints(directory: str = "."):
    """列出目录中的检查点文件"""
    checkpoint_files = []
    
    for file_path in Path(directory).glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'tasks' in data and 'stats' in data:
                    checkpoint_files.append(file_path)
        except:
            continue
    
    if not checkpoint_files:
        print(f"📂 在 {directory} 中未找到检查点文件")
        return
    
    print(f"📂 在 {directory} 中找到的检查点文件:")
    for file_path in checkpoint_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                tasks = data.get('tasks', [])
                completed = sum(1 for task in tasks if task.get('status') == 'completed')
                timestamp = data.get('timestamp', 'Unknown')
                
                print(f"  📁 {file_path.name}")
                print(f"     📊 {completed}/{len(tasks)} 完成 ({timestamp})")
        except:
            continue


def main():
    parser = argparse.ArgumentParser(description="检查点管理工具")
    parser.add_argument("command", choices=["status", "clean", "reset", "export", "list"],
                      help="操作命令")
    parser.add_argument("checkpoint_file", nargs='?', help="检查点文件路径")
    parser.add_argument("output_file", nargs='?', help="输出文件路径 (仅用于export)")
    parser.add_argument("--directory", "-d", default=".", help="目录路径 (仅用于list)")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_checkpoints(args.directory)
    elif args.command in ["status", "clean", "reset", "export"]:
        if not args.checkpoint_file:
            print("❌ 请指定检查点文件路径")
            return
            
        if args.command == "status":
            show_status(args.checkpoint_file)
        elif args.command == "clean":
            clean_checkpoint(args.checkpoint_file)
        elif args.command == "reset":
            reset_checkpoint(args.checkpoint_file)
        elif args.command == "export":
            output_file = args.output_file or f"checkpoint_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            export_report(args.checkpoint_file, output_file)


if __name__ == "__main__":
    main() 