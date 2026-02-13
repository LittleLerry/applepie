import json
import pandas as pd
import numpy as np
import argparse
from utils.helpers import run_tokenize_prompt_and_output, run_get_response_log_probs
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import os
import deepspeed
from deepspeed.accelerator import get_accelerator

def get_sft_dataset(path_to_sft_dataset: str):
    with open(path_to_sft_dataset, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame(data)
    ### run only for selected data
    df["rew"]
    return df["question"].to_list(), df["model_gen"].to_list()

class SFTDataset(Dataset):
    def __init__(self, data_dict):
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.response_mask = data_dict["response_mask"]
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "attention_mask": torch.ones_like(self.input_ids[idx]),  # Useless create attention mask
            "response_mask": self.response_mask[idx]
        }

def parse_args():
    parser = argparse.ArgumentParser(description='sft')

    parser.add_argument('--sft_data', type=str, default="/path/to/sft/data")
    parser.add_argument('--model_path', type=str, default="/path/to/model/to/be/sfted")
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=8)
    parser.add_argument('--sft_data_output', type=str, default="/path/to/tokenized/sft/data")
    parser.add_argument('--sft_ckpt_output_prefix', type=str, default="/path/to/save/model/ckpt")
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument(
        "--stage",
        default=2,
        type=int,
        choices=[0, 1, 2, 3],
        help="ZeRO stage",
    )

    # must for deepspeed
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    # must for warpper
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    return args

def train(args,engine,optimizer,trainloader):
    local_device = get_accelerator().device_name()
    local_rank = engine.local_rank

    for e in range(args.epoch):
        running_loss = 0.0
        engine.train()
        steps = 0
        for i, traindata_dict in enumerate(trainloader):
            steps = i
            input_ids = traindata_dict["input_ids"].to(local_device)
            labels = traindata_dict["labels"].to(local_device)
            response_mask = traindata_dict["response_mask"].to(local_device)
            
            log_probs_logits = run_get_response_log_probs(engine, input_ids,labels,False)["log_probs"]

            batch_size = input_ids.shape[0]
            # already backwarded
            token_loss = - log_probs_logits * response_mask
            total_loss = token_loss.sum() / batch_size
            running_loss = running_loss + total_loss

            engine.backward(total_loss)
            engine.step()

            if local_rank == 0 and i % args.log_interval == (args.log_interval - 1):
                print(f"[{e + 1 : d}, {i + 1 : 5d}] loss: {running_loss / args.log_interval : .8f}")
                running_loss = 0.0
            
            if (i in {128//args.batch_size, 256//args.batch_size, 512//args.batch_size, 1024//args.batch_size}):
                print(f"rank:{local_rank} saving after steps {steps} with total batch {steps*args.batch_size}\n")
                engine.save_checkpoint(save_dir=f"{args.sft_ckpt_output_prefix}{e+1}_step{i+1}.ckpt")
        print(f"rank:{local_rank} saving after steps {steps}\n")
        engine.save_checkpoint(save_dir=f"{args.sft_ckpt_output_prefix}{e+1}_step{i+1}.ckpt")

def check_dataset(path):
    return os.path.exists(path)


def get_ds_config(args):
    ds_config = {
        # It is effective training batch size. This is the amount of data samples that leads to one step of model update.
        # train_batch_size must be equal to train_micro_batch_size_per_gpu * gradient_accumulation_steps * number of GPUs
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": 1,
        "train_micro_batch_size_per_gpu": args.batch_size // args.gpus, # GPUs must be 4 in this case
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 4e-5,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 4e-5,
                "warmup_num_steps": 16,
            },
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        """
        #! idk wth is this...
        "bf16": {"enabled": args.dtype == "bf16"},
        "fp16": {
            "enabled": args.dtype == "fp16",
            "fp16_master_weights_and_grads": False,
            "loss_scale": 0,
            "loss_scale_window": 500,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "initial_scale_power": 15,
        },
        """
        "wall_clock_breakdown": False, # ?
        "zero_optimization": {
            "stage": args.stage,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 50000000,
            "reduce_bucket_size": 50000000,
            "overlap_comm": True,
            "contiguous_gradients": True,
            #"offload_optimizer": False,
        },
    }
    return ds_config


def main(args):
    if(args.local_rank == 0):
        print('Initializing the cluster')
    deepspeed.init_distributed()

    _local_rank = int(os.environ.get('LOCAL_RANK'))
    get_accelerator().set_device(_local_rank)
    # get dataset, batch size etc will be determined by some extral conf, so directly pass this dataset to the deepspeed is enough
    # SHIT CODE, will run on each process// FIXED on 1/26/2026


    # After half monthes of writing python code, now I still believe the following code is shit. // Comments on 2/13/2026
    # (1) Use single process the encode the entire training data to IDs, which is totaly not scaleable.
    # (2) Should spent more time making read those IDs faster!
    # (3) Use torch.distributed instead of deepspeed. Seems deepspeed is totally shit now, I even cannot replace attn impl.

    if torch.distributed.get_rank() == 0 and not check_dataset(args.sft_data_output):
        print("Tokenizing dataset.")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        p,r = get_sft_dataset(args.sft_data)
        data_dict = run_tokenize_prompt_and_output(prompt_strs=p,output_strs=r,tokenizer=tokenizer)
        torch.save(data_dict, args.sft_data_output)
    torch.distributed.barrier()
    # get dataset
    if torch.distributed.get_rank() == 0:
        print("Read sft_data_output and init dataset.")
    assert check_dataset(args.sft_data_output)
    data_dict = torch.load(args.sft_data_output)
    dataset = SFTDataset(data_dict)
    # get model
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    # get ds conf
    ds_config = get_ds_config(args)
    # launch engine
    engine, optimizer, trainloader, lr = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
        config=ds_config,
    )
    if(engine.local_rank == 0):
        print('Entrying the training function')
    train(args=args,engine=engine,optimizer=optimizer,trainloader=trainloader)

if __name__ == "__main__":
    """
    Training the qwen2.5 model use deepspeed on single node and 4/8 gpus after 1/2 epoch. 
    To eval the performance of the turned model, firstly convert it to hf tesnors using script
    generated by deepspeed, then replace the file model.safetensors in orignal qwen directory
    with those tensors, then can use vllm to launch it!
    """
    main(parse_args())