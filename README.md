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
nohup python evaluation/main.py --input_folder "/home/ubuntu/data/BigVideo-test/test" --output_path "/home/ubuntu/qwen_eval/evaluation/test_data/result" > /home/ubuntu/qwen_eval/output.log --use_audio &
```

__Recursive Translate__
``` bash
python evaluation/recursive_translate.py --input_folder "/home/ubuntu/data/DoveBench" --output_path "/home/ubuntu/qwen_eval/evaluation/test_data/dovebench_result" --use_audio 
# nohup会有bug，kill掉所有nohup进程
```


## Evaluation(using original repo and windows)
``` bash
scp -r ...@...:/home/ubuntu/qwen_eval/evaluation/test_data/result C:\TEMP2\coding\0_Recent\vidove\ViDove\evaluation\test_data\qwen_results
cd C:\TEMP2\coding\0_Recent\vidove\ViDove
python evaluation/generate_eval_result.py
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