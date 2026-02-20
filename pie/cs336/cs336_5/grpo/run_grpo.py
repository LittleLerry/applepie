import argparse
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import torch
from dataset import ListDataset, collate_strings
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import adapters
from drgrpo_grader import r1_zero_reward_fn
from unittest.mock import patch
import contextlib

def parse_args():
    parser = argparse.ArgumentParser(description='baseline')

    parser.add_argument('--model_path', type=str, default="./models")
    parser.add_argument('--train_path', type=str, default="./gsm8k/train.jsonl")
    parser.add_argument('--eval_path', type=str, default="./gsm8k/test.jsonl")
    parser.add_argument('--prompt_path', type=str, default="./r1_zero.prompt")

    parser.add_argument('--train_policy_device', type=str, default="cuda:1")
    parser.add_argument('--gradient_accumulation_step', type=int, default=16)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--rollout_batch_size', type=int, default=16)
    parser.add_argument('--group_size', type=int, default=8)
    parser.add_argument('--n_grpo_steps', type=int, default=3500)
    parser.add_argument('--epochs_per_rollout_batch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--advantage_eps', type=float, default=1e-6)
    parser.add_argument('--loss_type', type=str, default="grpo_clip")
    parser.add_argument('--use_std_normalization', type=bool, default=True)

    parser.add_argument('--cliprange', type=float, default=1.0)

    parser.add_argument('--old_policy_device', type=str, default="cuda:0")
    parser.add_argument('--sampling_temperature', type=float, default=1.0)
    parser.add_argument('--sampling_min_tokens', type=int, default=4)
    parser.add_argument('--sampling_max_tokens', type=int, default=1024)

    args = parser.parse_args()
    return args

# ===========================from assignment5 init code===========================
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    # vllm==0.7.2, if your vllm version is too high, vllm.worker will not be found
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
    # with world_size_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
# ===========================from assignment5 init code===========================
def load_policy_into_vllm_instance(args, policy, llm: LLM):
    state_dict = {k: v.to(args.old_policy_device) for k, v in policy.state_dict().items()}
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def init_dataloader(args):
    microbatch_size = args.train_batch_size // args.gradient_accumulation_step
    n_prompts_per_rollout = args.rollout_batch_size // args.group_size
    dataset = ListDataset(args.train_path, args.prompt_path)
    dataloader = DataLoader(dataset, batch_size= n_prompts_per_rollout , collate_fn=collate_strings)
    return dataloader

def val_reward(args, old_policy: LLM, eval_sampling_params, step):
    # with torch.inference_mode():
    group_size = 1
    questions, answers = ListDataset(args.eval_path, args.prompt_path).data
    repeated_questions = [item for item in questions for _ in range(group_size)]
    repeated_answers = [item for item in answers for _ in range(group_size)]
    outputs = old_policy.generate(repeated_questions, eval_sampling_params)
    rollout_responses = [output.outputs[0].text for output in outputs]
    _ , _ , metadata = adapters.run_compute_group_normalized_rewards(
                r1_zero_reward_fn, rollout_responses, 
                repeated_answers, group_size, args.advantage_eps, args.use_std_normalization)
    mr = metadata["mean_rewards"]
    print(f"step{step}_avg_eval_reward:{mr}")

def train(args):
    # check basic constraints
    assert args.rollout_batch_size % args.group_size == 0
    assert args.rollout_batch_size % args.train_batch_size == 0
    assert args.train_batch_size % args.gradient_accumulation_step == 0
    assert args.train_batch_size >= args.group_size

    print("Initializing old policy")
    old_policy = init_vllm(args.model_path, args.old_policy_device, 0)
    print("Initializing dataloader")
    dl = init_dataloader(args)
    print("Initializing policy")
    policy = AutoModelForCausalLM.from_pretrained(args.model_path, attn_implementation="flash_attention_2", dtype = torch.bfloat16).to(args.train_policy_device)
    optimizer = torch.optim.AdamW(policy.parameters(), 
                                  lr=args.learning_rate, 
                                  weight_decay=0.0,
                                  betas=(0.9, 0.95),
                                )
    eval_sampling_params = SamplingParams(temperature=args.sampling_temperature, 
                                          top_p=1.0, 
                                          max_tokens=args.sampling_max_tokens, 
                                          stop=["</answer>"], 
                                          min_tokens=args.sampling_min_tokens,
                                          include_stop_str_in_output = True
                                        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    scaler = GradScaler(args.train_policy_device)

    print("Entrying train loop")
    policy.train()
    for _ in range(1): # useless loop
        # n_grpo_steps loop
        for (idx, (questions, answers)) in enumerate(dl):
            if (idx >= args.n_grpo_steps):
                break
            # eval performance on eval set
            # SHIT DODE, not hide lag of inference
            if ((idx+1) % 20 == 0):
                print("Evaluating performance")
                val_reward(args, old_policy, eval_sampling_params, idx)

            # =========================Experience collection=========================
            # There are total rollout_batch_size // group_size questions = 256 // 8 = 32 questions to be rollouted.

            # Preparing and collecting rollout_batch_size questions and experiences from old policy
            repeated_questions = [item for item in questions for _ in range(args.group_size)]
            repeated_answers = [item for item in answers for _ in range(args.group_size)]
            #? LLM.generate can use strings in cpu, returning also in cpu.
            outputs = old_policy.generate(repeated_questions, eval_sampling_params)
            rollout_responses = [output.outputs[0].text for output in outputs]

            # response mask
            tokenized_qr = adapters.run_tokenize_prompt_and_output(repeated_questions, rollout_responses, tokenizer)
            response_mask = tokenized_qr["response_mask"]
            #response_mask = response_mask.to(args.train_policy_device)

            # advs
            advs, raw_rewards, metadata = adapters.run_compute_group_normalized_rewards(
                r1_zero_reward_fn, rollout_responses, 
                repeated_answers, args.group_size, args.advantage_eps, args.use_std_normalization)
            #advs = advs.to(args.train_policy_device)
            #raw_rewards = raw_rewards.to(args.train_policy_device)
            

            # update epochs_per_rollout_batch times, for each time, perform grad acc if necessary
            assert args.epochs_per_rollout_batch == args.rollout_batch_size // args.train_batch_size
            for j in range(args.epochs_per_rollout_batch):
                
                train_microbatch_size = args.train_batch_size // args.gradient_accumulation_step
                optimizer.zero_grad()

                avg_clip_ratio = 0.0
                avg_entrpy = 0.0

                for k in range(args.gradient_accumulation_step):
                    start = j * args.train_batch_size + k * train_microbatch_size
                    end = start + train_microbatch_size

                    with torch.inference_mode():
                        response_log_probs = adapters.run_get_response_log_probs(policy, tokenized_qr["input_ids"][start:end].to(args.train_policy_device), tokenized_qr["labels"][start:end].to(args.train_policy_device), True)
                        old_log_probs = response_log_probs["log_probs"]
                    
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                    # with contextlib.nullcontext():
                        response_log_probs_policy = adapters.run_get_response_log_probs(policy, tokenized_qr["input_ids"][start:end].to(args.train_policy_device), tokenized_qr["labels"][start:end].to(args.train_policy_device), True)
                        policy_log_probs = response_log_probs_policy["log_probs"]

                        loss, metadata  = adapters.run_grpo_microbatch_train_step(policy_log_probs, 
                                                                              response_mask[start:end].to(args.train_policy_device), 
                                                                              args.gradient_accumulation_step, args.loss_type, 
                                                                              raw_rewards[start:end].to(args.train_policy_device), 
                                                                              advs[start:end].to(args.train_policy_device), 
                                                                              old_log_probs, 
                                                                              args.cliprange, scaler)
                        
                        avg_clip_ratio += metadata.get("clip_ratio", 0.0)
                        avg_entrpy += (response_log_probs_policy["token_entropy"] * response_mask[start:end].to(args.train_policy_device)).sum() / response_mask[start:end].to(args.train_policy_device).sum()

                    del response_log_probs

                    # loss.backward()
                    # =======================For BFloat16 AMP, the grad scaler wouldn't be needed as its range is equivalent to fp32's.==============================
                    loss.backward()
                    # scaler.scale(loss).backward()
                
                avg_clip_ratio /= args.gradient_accumulation_step
                avg_entrpy /= args.gradient_accumulation_step
                print(f"step {idx}-{j}, avg_clip_ratio:{avg_clip_ratio}, avg_entrpy:{avg_entrpy}")
                # update policy
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                # scaler.step(optimizer)
                # scaler.update()

                optimizer.step()
            
            # replace old policy with updated one
            load_policy_into_vllm_instance(args, policy, old_policy)

def main(args):
    train(args)

if __name__ == "__main__":
# See: https://zhuanlan.zhihu.com/p/28177433249 
# • rollout_batch_size: 
#   - Unlike OpenRLHF, in the cs336 implementation, this parameter represents the 
#     total batch size AFTER rollout (not before). For each question, we generate 
#     `group_size` responses, there are `n_prompts_per_rollout_batc` questions per rollout, 
#     which produces a total of `rollout_batch_size` response. Sampling is handled by vLLM 
#     and may require multiple forward passes.
#
# • Experience collection:
#   - All `rollout_batch_size` responses are used to collect experience tuples
#     (rewards, etc.), which may also require multiple steps
#
# • train_batch_size:
#   - The collected experiences are trained in batches of size `train_batch_size`
#   - The model is updated after each batch.
#   - Total update steps per rollout: update_steps = rollout_batch_size // train_batch_size
#
# • Batch size relationship still holds:
#   micro_train_batch_size * grad_accumulation_steps * num_gpus = train_batch_size
#
# • On-policy vs. Off-policy:
#   - If update_steps > 1: Multiple gradient updates per rollout batch -> Off-policy
#     (the policy diverges from the behavior policy that generated the data)
#   - If update_steps == 1: One update per rollout -> On-policy
#     (model updates immediately after data collection, staying aligned with behavior policy)
#
# • Derived relationship:
#   epochs_per_rollout_batch = rollout_batch_size // train_batch_size
    main(parse_args())