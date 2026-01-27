import json
import pandas as pd
import numpy as np
import argparse
from tests.adapters import run_tokenize_prompt_and_output, run_sft_microbatch_train_step, run_get_response_log_probs
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import os
import deepspeed
from deepspeed.accelerator import get_accelerator

class INSTDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, prompt_path, seq_length, shuffle, train_bs):
        if tokenizer.eos_token_id is None or tokenizer.bos_token_id is None:
            raise ValueError("No EOS/BOS token for this tokenizer.")
        self.eos = tokenizer.eos_token_id
        self.train_bs = train_bs

        with open(prompt_path, 'r') as f:
            self.prompt_template = f.read()
        self.prompt_template = self.prompt_template.strip() # WTF idk why there is a \n at the end of THE BLOODYYYY FKINGGGGG FILE

        with open(dataset_path, 'r') as f:
            self.dataset = f.readlines()
            if shuffle:
                import random
                random.shuffle(self.dataset)

        self.seq_length = seq_length
            
        print("Tokenizing the inputs and adding eos...")
        ids = tokenizer(list(map(self.get_input_string, self.dataset)))["input_ids"]
        entire_input_ids = torch.cat(list(map(self.add_eos_token, ids)),dim=0)

        print("Preparing input_ids and labels...")
        self.total_batches = (entire_input_ids.shape[0] - 1) // self.seq_length
        entire_input_ids = entire_input_ids[: self.total_batches * self.seq_length + 1]
        self.input_ids = entire_input_ids[:-1].view(self.total_batches, -1)
        self.labels =entire_input_ids[1:].view(self.total_batches, -1)
        assert self.input_ids.shape[1] == self.labels.shape[1]            

    def __len__(self):
        # exactly same as drop_last
        return self.total_batches - (self.total_batches % self.train_bs)
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
        }
    def get_input_string(self, s):
        data = json.loads(s.strip())
        return self.prompt_template.replace("{response}",data["response"]).replace("{instruction}", data["prompt"])
    def add_eos_token(self, l):
        return torch.tensor(l + [self.eos])

def parse_args():
    parser = argparse.ArgumentParser(description='Instruction turning for llama3.1-8B')
    # dataset/conf related
    parser.add_argument('--model_path', type=str, default="/mnt/GPU_10T/models/meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument('--prompt_path', type=str, default="/home/?/cs/assignment5-alignment/cs336_alignment/prompts/alpaca_sft.prompt")
    parser.add_argument('--dataset_path', type=str, default="/home/?/cs/assignment5-alignment/data/sft/train.jsonl")
    parser.add_argument('--ckpt_dir', type=str, default="/mnt/GPU_10T/models/test_ckpt/")

    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--shuffle', type=bool, default=False)
    # model related 
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--log_interval', type=int, default=8)
    parser.add_argument('--train_micro_batch_size_per_gpu', type=int, default=2)
    parser.add_argument('--total_gpus', type=int, default=8)

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

    # There are 210348 lines in original train.jsonl. 
    for e in range(args.epoch):
        running_loss = 0.0
        engine.train()
        steps = 0
        for i, traindata_dict in enumerate(trainloader):
            steps = i
            input_ids = traindata_dict["input_ids"].to(local_device)
            labels = traindata_dict["labels"].to(local_device)

            # There is no mask for this dataloader, see the defination of class INSTDataset(Dataset)
            # get log probabilities given input_ids
            logits = engine(input_ids).logits
            log_probs = logits - torch.logsumexp(logits, dim=-1,keepdim=True)
            
            # get loss
            b, s = labels.shape
            negative_log_probs_of_labels = - log_probs[torch.arange(b)[:, None], torch.arange(s), labels]
            per_seq_loss = negative_log_probs_of_labels.sum() / b
            running_loss = running_loss + per_seq_loss

            # idiot backward
            engine.backward(per_seq_loss)
            engine.step()

            if local_rank == 0 and i % args.log_interval == (args.log_interval - 1):
                print(f"[{e + 1 : d}, {i + 1 : 5d}] loss: {running_loss / args.log_interval : .8f}")
                running_loss = 0.0

        print(f"rank:{local_rank} saving after steps {steps}\n")
        engine.save_checkpoint(save_dir=f"{args.ckpt_dir}{e}_{steps}ckpt")


def get_ds_config(args, total_steps):
    ds_config = {
        # It is effective training batch size. This is the amount of data samples that leads to one step of model update.
        # train_batch_size must be equal to train_micro_batch_size_per_gpu * gradient_accumulation_steps * number of GPUs
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.batch_size // (args.train_micro_batch_size_per_gpu * args.total_gpus),
        "train_micro_batch_size_per_gpu": args.train_micro_batch_size_per_gpu, # 
        "steps_per_print": 1,

        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 2e-5,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7,
            },
        },

        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": total_steps,
                "warmup_min_ratio": 0,
                "warmup_num_steps": int(total_steps * 0.03),
                "cos_min_ratio": 0.0001,
            },
        },
        
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        # bf16 settings
        "bf16": {
            "enabled": True,
        },

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
    # ALSO shit code: WHY NOT USE C_FN???????
    # if torch.distributed.get_rank() == 0 and not check_dataset(args.sft_data_output):
    #    print("Tokenizing dataset.")
    #    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    #    p,r = get_sft_dataset(args.sft_data)
    #    data_dict = run_tokenize_prompt_and_output(prompt_strs=p,output_strs=r,tokenizer=tokenizer)
    #    torch.save(data_dict, args.sft_data_output)
    # torch.distributed.barrier()
    # get dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = INSTDataset(tokenizer, args.dataset_path, args.prompt_path, args.seq_len, args.shuffle, args.batch_size)
    torch.distributed.barrier()

    assert len(dataset) % args.batch_size == 0
    total_steps = (len(dataset) // args.batch_size)

    if torch.distributed.get_rank() == 0:
        print(f"All ranks have parsed their input_ids. total steps: {total_steps}. (BAD CODE, FIX LATTER)\n") #3363
    
    # get model
    if torch.distributed.get_rank() == 0:
        print(f"Loading model and confs\n")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    ds_config = get_ds_config(args, total_steps)

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
    # deepspeed ./run_instruction_turning.py --deepspeed

    main(parse_args())
