
from model import BasicsTransformerLM
from optimizer import AdamW
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def get_non_random_train_data(rank:int, world_size:int, conf:dict):
    batch_size = 1
    seq = conf["context_length"]

    vocab_size = conf["vocab_size"]
    tile_size = batch_size * seq

    if rank >= 0:
        #print(f"T:{torch.arange(tile_size*rank, tile_size*(rank+1)).view(batch_size, seq) % vocab_size}")
        yield torch.arange(tile_size*rank, tile_size*(rank+1)).view(batch_size, seq) % vocab_size
    # rank < 0 preserved for globol training
    else:
        #print(f"R:{torch.arange(0, tile_size*world_size).view(batch_size * world_size, seq) % vocab_size}")
        yield torch.arange(0, tile_size*world_size).view(batch_size * world_size, seq) % vocab_size

def train(rank: int, world_size: int, conf: dict, epoch: int, untrained_para: str, trained_para: str): 
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # init model for each process
    device = f"cuda:{rank}"
    torch.set_default_device(device)
    model = BasicsTransformerLM(**conf).to(device)
    # broadcast model's and opt's parameters from rank 0 to all other ranks
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    print("synced model state")
    opt = AdamW(model.parameters(), lr= 0.001) # large lr to verify correctness, should we boardcast it???????????????????????????????:
    # safe arrival point
    torch.cuda.synchronize()
    dist.barrier()
    if (rank == 0):
        print("model paras shared. Entrying training loop.")
    # train loop
    model.train()

    # no warm up timer
    acc = torch.tensor(0.0)
    e_start = time.time()
    for _ in range(epoch):
        for (_, data) in enumerate(get_non_random_train_data(rank, world_size, conf)):
            # forward
            opt.zero_grad()
            input = data.to(device)
            output = model(input)
            loss = output.sum()
            loss = loss / 1.0 # batch size

            loss.backward()

            # sync grads
            sync_grad_start = time.time()
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)
                    param.grad = param.grad / (world_size)
            torch.cuda.synchronize()
            d = time.time() - sync_grad_start
            acc += d

            opt.step()
    # arrival
    torch.cuda.synchronize()
    dist.barrier()
    e_end = time.time()
    duration = torch.tensor(e_end - e_start)

    dist.all_reduce(duration)
    dist.all_reduce(acc)

    if (rank == 0):
        print(f"Training for the model has been done. avg_t:{duration/world_size}, avg_sync_t:{acc/world_size}, sync_t/t={100 * acc / duration}%")
    dist.barrier()
    torch.distributed.destroy_process_group()


"""
================================================benchmark result==========================================================
conf = {"d_model":1600, "d_ff": 6400, "num_layers" : 48, "num_heads": 25, "vocab_size": 10000, "context_length": 128, "rope_theta": 0.5}

Training for the model has been done. avg_t:7.386388301849365, avg_sync_t:0.6150552034378052, sync_t/t=8.326873779296875%
"""
if __name__ == "__main__":
    if not torch.cuda.is_available():
        exit(0)
    print("launching dist.")
    conf = {"d_model":1600, "d_ff": 6400, "num_layers" : 48, "num_heads": 25, "vocab_size": 10000, "context_length": 128, "rope_theta": 0.5}
    # conf = {"d_model":768, "d_ff": 3072, "num_layers" : 12, "num_heads": 12, "vocab_size": 1000, "context_length": 128, "rope_theta": 0.5} # toy for testing

    world_size = 8
    untrained_para_path = "./untrained_"
    trained_para_path = "./trained_"
    epoch = 16
    # init
    # def train(rank: int, world_size: int, conf: dict, epoch: int, untrained_para: str, trained_para: str): 
    mp.spawn(fn=train, args=(world_size, conf, epoch, untrained_para_path, trained_para_path), nprocs=world_size, join=True)