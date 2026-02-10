
from model import BasicsTransformerLM
from optimizer import AdamW
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

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
    # save 
    if rank == 0:
        torch.save(model.state_dict(), untrained_para+f"model")
        torch.save(opt.state_dict(), untrained_para+f"opt")
    # safe arrival point
    torch.cuda.synchronize()
    dist.barrier()
    if (rank == 0):
        print("model paras shared. Entrying training loop.")

    # train loop
    model.train()
    for _ in range(epoch):
        for (_, data) in enumerate(get_non_random_train_data(rank, world_size, conf)):
            # forward
            opt.zero_grad()
            input = data.to(device)
            output = model(input)
            loss = output.sum()
            loss = loss / 1.0 # batch size

            # =====================debug only============================
            # losses = [torch.zeros_like(loss) for _ in range(world_size)]
            # dist.all_gather(losses, loss)
            # print(f"Trained{epoch}:\t{torch.tensor(losses).mean()}")

            loss.backward()

            # sync grads
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)
                    param.grad = param.grad / (world_size)

            opt.step()
    # save
    dist.barrier()
    if (rank == 0):
        print("Training for the model has been done.")
        torch.save(model.state_dict(), trained_para+f"model")
        torch.save(model.state_dict(), trained_para+f"opt")
    dist.barrier()
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    # NCCL_DEBUG=WARN python native_ddp.py
    if not torch.cuda.is_available():
        exit(0)
    print("launching dist.")
    conf = {"d_model":2560, "d_ff": 10240, "num_layers" : 32, "num_heads": 32, "vocab_size": 10000, "context_length": 128, "rope_theta": 0.5}
    # conf = {"d_model":768, "d_ff": 3072, "num_layers" : 12, "num_heads": 12, "vocab_size": 1000, "context_length": 128, "rope_theta": 0.5} # toy for testing

    world_size = 2
    untrained_para_path = "./untrained_"
    trained_para_path = "./trained_"
    epoch = 1024
    # init
    # def train(rank: int, world_size: int, conf: dict, epoch: int, untrained_para: str, trained_para: str): 
    mp.spawn(fn=train, args=(world_size, conf, epoch, untrained_para_path, trained_para_path), nprocs=world_size, join=True)
    print("loading ref")
    ref_model = BasicsTransformerLM(**conf).cuda()
    ref_model.load_state_dict(torch.load(untrained_para_path+f"model"))
    ref_opt = AdamW(ref_model.parameters(), lr= 0.001)
    ref_opt.load_state_dict(torch.load(untrained_para_path+f"opt"))

    ref_model.train()
    for _ in range(epoch):
        for (_, data) in enumerate(get_non_random_train_data(-1, world_size, conf)):
            # forward
            ref_opt.zero_grad()
            input = data.cuda()
            output = ref_model(input)
            loss = output.sum() / (1.0 * world_size)# batch_size * world_size

            # =====================debug only============================
            # print(f"Retrained{epoch}:\t{loss}")

            loss.backward()

            ref_opt.step()
    
    model = BasicsTransformerLM(**conf).cuda()
    model.load_state_dict(torch.load(trained_para_path+f"model"))
    abs_sum_error = 0.0

    for param_a, param_b in zip(model.parameters(), ref_model.parameters()):
        abs_sum_error += torch.abs(param_a.data - param_b.data).sum()
    print("relative abs error:"+str(abs_sum_error/ sum(p.numel() for p in model.parameters())))
    # relative abs error:tensor(0.0002, device='cuda:0')
