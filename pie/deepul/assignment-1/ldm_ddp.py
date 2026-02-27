# latant diffusion model use vq-vae in celeha-hqr dataset
"""
implmente the arch is simple, but the hyperpara turning can be shit. 

"""
import torch
import torch.nn as nn
import random
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from tqdm import tqdm

# distributed settings
def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
def cleanup():
    dist.destroy_process_group()

# dataset AI generated code
class celeba_hqr(Dataset):
    def __init__(self, cache_file_name):
        assert os.path.exists(cache_file_name)
        self.data = torch.load(cache_file_name, weights_only=True)
        self.l = len(self.data)

    def __len__(self):
        return self.l
    
    def __getitem__(self, idx):
        return (self.data[idx], torch.tensor(0, dtype= torch.int32)) # ((C, H, W), l)
# AI generated code
def save_tensor_to_image(tensor, save_path, image_name):
    os.makedirs(save_path, exist_ok=True)

    tensor = tensor.detach().cpu()
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor.permute(1, 2, 0)

    numpy_img = tensor.numpy()
    numpy_img = (numpy_img * 255).astype(np.uint8)

    pil_img = Image.fromarray(numpy_img, mode='RGB')
    full_path = os.path.join(save_path, image_name)

    pil_img.save(full_path)

# model definition
class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.l1 = nn.Linear(in_features, 2 * out_features)
        self.l2 = nn.Linear(2 * out_features, out_features)
        self.silu = nn.SiLU()
    def forward(self, x):
        # (, d_ty) -> (, d_channel)
        x = self.l1(x)
        x = self.silu(x)
        return self.l2(x)

class residual_layer(nn.Module):
    def __init__(self, in_channels, d_t, d_y):
        # (*, C, H, W)
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.silu = nn.SiLU()
        self.bn1 = nn.BatchNorm2d(num_features= in_channels)

        self.mlpt = MLP(d_t, in_channels) # (in_channels, ) not shared?
        self.mlpy = MLP(d_y, in_channels) # (in_channels, ) not shared?

        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_features= in_channels)
    
    
    def forward(self, x, t, y):
        # x: (*,C,H,W)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)
        out = out + self.mlpt(t).unsqueeze(-1).unsqueeze(-1) + self.mlpy(y).unsqueeze(-1).unsqueeze(-1)

        out = self.conv2(out)
        out = self.bn2(out)
        
        return x + self.silu(out)

class encoder(nn.Module):
    def __init__(self, in_channels, d_t, d_y):
        super().__init__()
        self.res_layers = nn.Sequential(*[residual_layer(in_channels, d_t, d_y) for _ in range(2)]) # [residual_layer(in_channels, d_t, d_y) for _ in range(2)] will not move to cuda automatically!
        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x, t, y):
        for layer in self.res_layers:
            x = layer(x, t, y)
        return self.down(x)

class midcoder(nn.Module):
    def __init__(self, in_channels, d_t, d_y):
        super().__init__()
        self.bottleneck = nn.Sequential(*[residual_layer(in_channels, d_t, d_y) for _ in range(3)])

    def forward(self, x, t, y):
        for layer in self.bottleneck:
            x = layer(x, t, y)
        return x

class decoder(nn.Module):
    def __init__(self, in_channels, d_t, d_y):
        super().__init__()
        self.res_layers = nn.Sequential(*[residual_layer(in_channels, d_t, d_y) for _ in range(2)])
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x, t, y):
        x = self.up(x)
        for layer in self.res_layers:
            x = layer(x, t, y)
        return x

class Unet(nn.Module):
    def __init__(self, input_channel, d_model, empty_condition_rate, num_conditions, total_steps):
        super().__init__()

        # t is float within range [0,1]
        # y is label here (also can be other domain experts' output embeddings like CLIP's output)
        
        # self.emd_t_size = 16
        # self.emd_t = nn.Linear(1, self.emd_t_size) # may cause issue here


        self.emd_t_size = 512
        self.emd_t = nn.Embedding(total_steps, self.emd_t_size)
        self.total_steps = total_steps

        self.emd_y_size = 512
        assert num_conditions >= 0
        self.y_range = num_conditions
        self.emd_y = nn.Embedding(num_conditions + 1, self.emd_y_size)
        self.empty_condition_rate = empty_condition_rate
        self.d_model = d_model

        self.init_conv = nn.Conv2d(input_channel, d_model, 3,1,1)

        self.e1 = encoder(d_model, self.emd_t_size, self.emd_y_size)
        self.e2 = encoder(d_model, self.emd_t_size, self.emd_y_size)
        self.e3 = encoder(d_model, self.emd_t_size, self.emd_y_size)
        self.m1 = midcoder(d_model, self.emd_t_size, self.emd_y_size)
        self.d3 = decoder(d_model, self.emd_t_size, self.emd_y_size)
        self.d2 = decoder(d_model, self.emd_t_size, self.emd_y_size)
        self.d1 = decoder(d_model, self.emd_t_size, self.emd_y_size)

        self.final_conv = nn.Conv2d(d_model, input_channel, 3,1,1)
    
    def forward(self, input, timestamp, condition):
        # input (*, C, H, W)
        # timestamp (*) -> (*, d_t)
        # condition (*) -> (*, d_y)
        # assert for all condition < self.y_range

        timestamp = torch.round(timestamp * self.total_steps).clamp(0, self.total_steps -1).to(device=input.device, dtype=torch.int64)
        t = self.emd_t(timestamp) # (*, d_t)

        # replace with empty condition randomly
        mask = torch.rand(condition.shape, device=t.device) < self.empty_condition_rate
        condition[mask] = self.y_range
        y = self.emd_y(condition)

        x = self.init_conv(input) # (*, d_model, H, W)
        
        o1 = self.e1(x, t, y)
        o2 = self.e2(o1, t, y)
        o3 = self.e3(o2, t, y) # TORCH_DISTRIBUTED_DEBUG=DETAIL python ldm_ddp.py, I misswrite self.e2(o2, t, y)

        o4 = self.m1(o3, t, y)

        o5 = self.d3(o3+o4,t,y)
        o6 = self.d2(o2+o5,t,y)
        o7 = self.d1(o1+o6,t,y)

        # output (*, C, H, W)
        return self.final_conv(o7)

def train(rank, conf):
    # init ddp confs
    world_size = conf["world_size"]
    local_rank = rank # single node
    torch.cuda.set_device(local_rank)
    setup(rank, world_size, "nccl")
    if (rank == 0):
        print("cluster inited")
    device = torch.device("cuda", local_rank)

    # load model confs
    d_model = conf["d_model"]
    empty_rate = conf["empty_rate"]
    num_labels = conf["num_labels"]
    channels = conf["channels"]
    train_data = conf["train_cache"]
    eval_data = conf["eval_cache"]
    batch_size = conf["batch_size"]
    epoch = conf["epoch"]
    lr = conf["lr"]
    # eval_epoch_interval = conf["eval_epoch_interval"]
    sample_epoch_interval = conf["sample_epoch_interval"] 
    num_samples = conf["num_samples_per_gpu"]
    width = conf["width"]
    steps = conf["steps"]
    guidance_scale = conf["guidance_scale"]
    save_epoch_interval = conf["save_epoch_interval"]
    sampling_output_dir = conf["sampling_output_dir"]

    # init model etc.
    ddp_model = DDP(Unet(channels, d_model, empty_rate, num_labels, steps).to(device), device_ids=[local_rank], output_device=local_rank)
    empty_token_id = num_labels # list int: [0,1,...,num_labels-1, num_labels], where num_labels is id indicates that it is empty condition

    train_dataset = celeba_hqr(train_data)
    # eval_dataset = celeba_hqr(eval_data)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=3, shuffle=(train_sampler is None))

    # eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank)
    # eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, sampler=eval_sampler, num_workers=3, shuffle=(eval_sampler is None))

    opt = torch.optim.AdamW(ddp_model.parameters(), lr=lr)
    slr = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epoch)
    criterion = nn.MSELoss(reduction='none')

    # training
    if(rank == 0):
        print("entry training loop")
    ddp_model.train()
    

    for e in range(epoch):
        train_sampler.set_epoch(e)
        for idx, (images, labels) in enumerate(train_dataloader):
            #=========Classifier-free guidance training for Gaussian probability=========
            shape = images.shape[:-3] # (*, )
            z, y = images.to(device), labels.to(device) # (*, C, H, W) and (*,)

            t = torch.rand(size=shape, device=device) # (*, )
            noise = torch.randn_like(z, device=device) # (*, C, H, W)
            
            alpha_t = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # (*, 1, 1, 1)
            beta_t = 1 - alpha_t # (*, 1, 1, 1)
            dalpha_t = torch.ones_like(alpha_t, device=device) * 1.0 # (*, 1, 1, 1)
            dbeta_t = torch.ones_like(alpha_t, device=device) * -1.0 # (*, 1, 1, 1)

            # input, timestamp, condition

            loss = criterion(ddp_model(alpha_t * z + beta_t * noise, t, y) , (dalpha_t * z + dbeta_t * noise)).view(*shape, -1).sum(-1).mean() # critial! 
            
            loss.backward()
            opt.step()
            opt.zero_grad()
            # log
            dist.reduce(loss, 0)
            if (rank == 0):
                print(f"epoch {e}, step {idx}, avg_loss {loss / world_size}")

        # sampling
        if ((e) % sample_epoch_interval == 0):
            with torch.inference_mode():
                """
                def forward(self, input, timestamp, condition):
                # input (*, C, H, W)
                # timestamp (*) -> (*, d_t)
                # condition (*) -> (*, d_y)
                # assert for all condition < self.y_range
                """
                t = torch.zeros(size=(num_samples,), device=device, dtype=torch.int64) # timestamps
                h = 1 / steps
                samplings = torch.randn(size=(num_samples, channels, width, width), device=device)
                
                y = torch.zeros(size=(num_samples,), device=device, dtype=torch.int64) # 0 labels

                empty_y = torch.zeros(size=(num_samples,), device=device, dtype=torch.int64) + empty_token_id

                # simulation can takes at different acc, here we use same acc as training
                for _ in range(steps):
                    u_empty = ddp_model(samplings, t, empty_y) # (*, C, H, W)
                    u = ddp_model(samplings, t, y) # (*, C, H, W)
                    u_hat = (1 - guidance_scale) * u_empty + guidance_scale * u
                    samplings = samplings + h * u_hat
                    t = t + h
                # collecting results
                if rank == 0:
                    gather_list = [torch.zeros_like(samplings, device=device) for _ in range(world_size)]
                    label_list = [torch.zeros_like(y, device=device) for _ in range(world_size)]
                else:
                    gather_list = None
                    label_list= None
                dist.gather(samplings, gather_list, dst=0)
                dist.gather(y, label_list, dst=0)

                if rank == 0:
                    result = torch.cat(gather_list, dim=0)
                    l = torch.cat(label_list, dim=0)
                    for i in range(result.shape[0]):
                        save_tensor_to_image(result[i], sampling_output_dir+f"/epoch{e}", f"{i}_{l[i]}.png")
        # save
        if ((e) % save_epoch_interval == 0) and (rank == 0):
            torch.save(ddp_model.module.state_dict(), f"./models/{e}.pt")
        slr.step()
        dist.barrier()
        # ===== done epoch loop ======
    cleanup()

def init_dataset_cache(root_dir, width, cache_file_name):
    if os.path.exists(cache_file_name):
        return

    data = [] 
    transform = transforms.Compose([
            transforms.CenterCrop(width),        # Central clip
            transforms.RandomHorizontalFlip(p=0.5), # Random mirror (p=1.0 ensures it applies logic)
            transforms.Resize((width, width)),   # Resize to given size
            transforms.ToTensor()                # Convert to Tensor (C, H, W) and scales accordingly
    ])
    image_paths = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        
    for file_name in os.listdir(root_dir):
        if file_name.lower().endswith(valid_extensions):
            image_paths.append(os.path.join(root_dir, file_name))

    image_paths.sort()
    for path in tqdm(image_paths, desc="Processing images"):
        try:
            img = Image.open(path).convert('RGB')
            processed_img = transform(img)
            data.append(processed_img)
        except Exception as e:
            pass
    
    print("stacking cache")
    stacked_tensor = torch.stack(data) 
    print("writing cache")
    torch.save(stacked_tensor, cache_file_name)

def get_conf():
    """
    (width, total_batch_size, d_model, lr) = (256, 64, 128, 8e-4) after ??? epoches with loss ~= ???
    (width, total_batch_size, d_model, lr) = (256, 128, 128, 8e-4) after ??? epoches with loss ~= ???
    (width, total_batch_size, d_model, lr) = (256, 128, 128, 4e-4) after ??? epoches with loss ~= ???

    (width, total_batch_size, d_model, lr) = (256, 64, 256, 8e-4) after ??? epoches with loss ~= ???
    (width, total_batch_size, d_model, lr) = (256, 128, 256, 1e-4) after ??? epoches with loss ~= ???
    (width, total_batch_size, d_model, lr) = (256, 128, 512, 1e-4) after ??? epoches with loss ~= ??? *****

    (width, total_batch_size, d_model, lr) = (64, 64, 512, 4e-4) after ??? epoches with loss ~= ???

    (width, total_batch_size, d_model, lr) = (256, 128, 256, 8e-4) after ??? epoches with loss ~= ???
    """

    conf = {
        "train_data_path": "./celeba_hqr/train",
        "eval_data_path" : "./celeba_hqr/eval",
        "width": 256,
        "train_cache" : "./t.pt",
        "eval_cache" : "./e.pt",
        "batch_size" :8,
        "d_model": 512,
        "empty_rate":0.1,
        "num_labels":0,
        "channels": 3,
        "epoch": 256,
        "lr": 1e-4,
        "eval_epoch_interval" : 16,
        "sample_epoch_interval" : 16,
        "save_epoch_interval": 16,
        "num_samples_per_gpu": 2,
        "steps": 256,
        "guidance_scale": 1.0,
        "sampling_output_dir": "./ldm_output",
        "world_size": 8,
    }
    return conf



if __name__ == '__main__':
    # nvidia only
    if not torch.cuda.is_available():
        print("Fuck.")
        exit(0)

    # necessary confs
    conf = get_conf()

    # preparing dataset
    init_dataset_cache(conf["train_data_path"], conf["width"], conf["train_cache"])
    init_dataset_cache(conf["eval_data_path"], conf["width"], conf["eval_cache"])

    # launching training
    print("init cluster")
    mp.spawn(fn=train, args=(conf, ), nprocs=conf["world_size"], join=True) # To be trained on 8*H800
