from typing import Callable, List, Dict, Tuple, Set, Optional, Union, Any
import pandas as pd
import json
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import argparse
from vllm import LLM, SamplingParams
import os

def evaluate_vllm(
        vllm_model: LLM,
        reward_fn: Callable[[str, str], dict[str, float]],
        prompts: List[str],
        answers: List[str],
        eval_sampling_params: SamplingParams,
    ):
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    generated_text = [output.outputs[0].text for output in outputs]

    #for i in range(len(generated_text)):
    #    print(f"g{i}:{generated_text[i]}")

    assert len(generated_text) == len(answers)
    return generated_text, [reward_fn(a, b) for a, b in zip(generated_text, answers)]

def extract_answer_from_gsm8k(answer_string):
    sp = "####"
    last_index = answer_string.rfind(sp)
    if last_index == -1:
        raise ValueError("Bad answer:" + answer_string)
    return answer_string[last_index + len(sp):].strip()


def replace_target_in_file(question_string, replacement):
    return question_string.replace('{question}', replacement)


def get_dataset(path_to_dataset: str, path_to_template: str):
    with open(path_to_dataset, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    with open(path_to_template, 'r', encoding='utf-8') as f:
        prompt = f.read()

    df = pd.DataFrame(data)
    return df['question'].apply(lambda s: replace_target_in_file(prompt,s)).tolist(), df['answer'].apply(lambda s: extract_answer_from_gsm8k(s)).tolist()

def save_to_jsonl(q,a,g,r,path):
    df = pd.DataFrame({
    'question': q,
    'answer': a, 
    'model_gen': g, 
    'reward': r
    })
    df.to_json(path, orient='records', lines=True, force_ascii=False)

def save_metric(r,path):
    total_format_reward = 0.0
    total_answer_reward = 0.0
    total_reward = 0.0
    f1a1 = 0
    f1a0 = 0
    f0a1 = 0
    f0a0 = 0
    for record in r:
        total_format_reward = total_format_reward + record["format_reward"]
        total_answer_reward = total_answer_reward + record["answer_reward"]
        total_reward = total_reward + record["reward"]
        if (record["format_reward"] == 1.0 and record["answer_reward"] == 1.0):
            f1a1 = f1a1 + 1
        if (record["format_reward"] == 1.0 and record["answer_reward"] == 0.0):
            f1a0 = f1a0 + 1
        if (record["format_reward"] == 0.0 and record["answer_reward"] == 1.0):
            f0a1 = f0a1 + 1
        if (record["format_reward"] == 0.0 and record["answer_reward"] == 0.0):
            f0a0 = f0a0 + 1
    with open(path, 'w') as f:
        f.write(f"avg_format_reward = {total_format_reward/len(r)}\n")
        f.write(f"avg_answer_reward = {total_answer_reward/len(r)}\n")
        f.write(f"avg_reward = {total_reward/len(r)}\n")
        f.write(f"f1a1 = {f1a1}\n")
        f.write(f"f1a0 = {f1a0}\n")
        f.write(f"f1a0 = {f0a1}\n")
        f.write(f"f0a0 = {f0a0}\n")

"""
 Summing the number of blue and white bolts </think> <answer> 3 </answer>
"""

def parse_args():
    parser = argparse.ArgumentParser(description='baseline')

    parser.add_argument('--obtain_sft_data', type=bool, default=False)
    parser.add_argument('--input_dataset', type=str, default="./data/gsm8k/test.jsonl")
    parser.add_argument('--input_prompt', type=str, default="./cs336_alignment/prompts/r1_zero.prompt")
    parser.add_argument('--model_path', type=str, default="/home/zzx/models/Qwen/Qwen2.5-Math-1.5B")

    args = parser.parse_args()
    return args


def main(args):
    q, a = get_dataset(args.input_dataset,args.input_prompt)
    model_name = args.model_path.split('/')[-1]

    sampling_parames =SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], min_tokens=4,include_stop_str_in_output = True)

    tensor_parallel_size = 1
    gpu_memory_utilization = 0.5
    # Example usage: CUDA_VISIBLE_DEVICES=devices TIKTOKEN_ENCODINGS_BASE=path/to/oss/encoder python baseline.py --obtain_sft_data True --input_dataset ./data/gsm8k/train.jsonl --model_path /mnt/GPU_10T/models/openai-mirror/gpt-oss-120b
    # CUDA_VISIBLE_DEVICES=4,5,6,7 TIKTOKEN_ENCODINGS_BASE=/home/zzx/model_distillation/my_test/oss_encoder python baseline.py --obtain_sft_data True --input_dataset ./data/gsm8k/train.jsonl --model_path /mnt/GPU_10T/models/openai-mirror/gpt-oss-120b
    if(args.obtain_sft_data):
        devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if (devices is None):
            raise Exception("NO CUDA VISIBLE FOUND.")
        tensor_parallel_size = len(devices.split(","))
        print(f"Using {devices}.")
        gpu_memory_utilization = 0.9
    
    llm = LLM(model=args.model_path, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization = gpu_memory_utilization, trust_remote_code=True)

    g, r = evaluate_vllm(llm,r1_zero_reward_fn,q,a,sampling_parames)
    save_metric(r,f'./baseline_{model_name}.txt')
    save_to_jsonl(q,a,g,r,f'./baseline_{model_name}.jsonl')

if __name__ == "__main__":
    main(parse_args())