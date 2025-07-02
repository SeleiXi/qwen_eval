original repo: https://github.com/pigeonai-org/ViDove

## Get Started

``` bash
conda create -n ViDove python=3.10 -y
conda activate ViDove
pip install --upgrade pip
pip install -r requirements.txt
```


## Run

__Main.py__
``` bash
nohup python evaluation/main.py --input_folder "/home/ubuntu/data/BigVideo-test/test" --output_path "/home/ubuntu/qwen_eval/evaluation/test_data/main_result" > /home/ubuntu/qwen_eval/output.log --use_audio &
```

__Recursive Translate__
``` bash
python evaluation/recursive_translate.py --input_folder "/home/ubuntu/data/DoveBench" --output_path "/home/ubuntu/qwen_eval/evaluation/test_data/dovebench_result" --use_audio 
# nohup会有bug，kill掉所有nohup进程
```


### most recent

reversed_version
``` bash
python evaluation/main_reverse.py --input_folder "/home/ubuntu/data/BigVideo-test/test" --output_path "/home/ubuntu/qwen_eval/evaluation/test_data/main_result"  --parallel_workers 20 --shared_model --use_audio
```


__batch processing Main__
```bash
python evaluation/batch_parallel_translate.py --input_folder "/home/ubuntu/data/BigVideo-test/test" --output_path "/home/ubuntu/qwen_eval/evaluation/test_data/main_result" --shared_model --use_audio --parallel_workers 50
```


__batch processing recursive__
```bash
python evaluation/recursive_parallel_translate.py --input_folder "/home/ubuntu/data/DoveBench" --output_path "/home/ubuntu/qwen_eval/evaluation/test_data/dovebench_result" --use_audio --parallel_workers 32 --shared_model

nohup python evaluation/recursive_parallel_translate.py --input_folder "/home/ubuntu/data/DoveBench" --output_path "/home/ubuntu/qwen_eval/evaluation/test_data/dovebench_result" --use_audio --parallel_workers 32 --shared_model &
```

## ✨ 性能优化版本 (推荐) 

**新增高性能版本**: `optimized_parallel_translate.py` - 解决了原版本运行慢、不稳定和无中间状态保存的问题

### 主要优化特性
- 🔄 **断点续传**: 程序中断后自动从上次停止处继续
- 💾 **中间状态保存**: 每30秒自动保存进度，避免重复处理  
- 🚀 **性能提升50%**: 优化GPU使用、去除复杂分段逻辑
- 📊 **实时监控**: 详细的进度显示和性能统计
- 🛡️ **错误恢复**: 单个文件失败不影响整体进程
- ✅ **稳定性增强**: 修复pickle错误，简化架构提高成功率

### 推荐使用命令 (已修复稳定性问题)

```bash
# 高性能批处理 (推荐 - 最新修复版本)
python evaluation/optimized_parallel_translate.py \
  --input_folder "/home/ubuntu/data/DoveBench" \
  --output_path "/home/ubuntu/qwen_eval/evaluation/test_data/optimized_results" \
  --checkpoint_file "./progress_checkpoint.json" \
  --parallel_workers 12 \
  --max_concurrent_gpu 3 \
  --disable_segmentation \
  --preserve_structure \
  --use_audio

# 稳定配置 (适合有问题的视频文件)
python evaluation/optimized_parallel_translate.py \
  --input_folder "/home/ubuntu/data/DoveBench" \
  --output_path "/home/ubuntu/qwen_eval/evaluation/test_data/dovebench_stable" \
  --checkpoint_file "./dovebench_checkpoint.json" \
  --parallel_workers 6 \
  --max_concurrent_gpu 2 \
  --disable_segmentation \
  --use_audio
```

> 📝 详细说明请参考: [PERFORMANCE_OPTIMIZATION.md](evaluation/PERFORMANCE_OPTIMIZATION.md)
> 🔧 **最新更新**: 修复了pickle错误，简化了代码架构，显著提高了处理成功率





## Evaluation(using original repo and windows)
``` bash
scp -r ...@...:/home/ubuntu/qwen_eval/evaluation/test_data/result C:\TEMP2\coding\0_Recent\vidove\ViDove\evaluation\test_data\qwen_results
cd C:\TEMP2\coding\0_Recent\vidove\ViDove
python evaluation/generate_eval_result.py
python evaluation/utils/identify_dirty_data.py
python evaluation/evaluate.py # 获得一个qwen_result.csv
python evaluation/utils/cal_avg_scores_from_results.py
```

## Evaluation(using server)
``` bash
git clone ... --branch Karsa/Streamlit-Migration
# 改generate_eval_result里面的路径为实际要eval的文件路径
python evaluation/generate_eval_result.py # 后面的给本地跑就好了，这样就只需要改generate_eval_result里面的路径就可以了，不那么麻烦
scp -r ...
```