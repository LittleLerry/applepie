import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from tqdm import tqdm
"""

Originally, I considered this task to be relatively straightforward, and it appeared to function adequately on a single GPU. 
However, when scaling to eight GPUs, some optimizations were necessary to achieve improved training performance.

In the context of the original dataset, it was observed that employing list indexing within the __getitem__ method resulted in 
underutilization of the eight GPUs. They are starving. It is also advisable to increase the number of workers in the DataLoader. 
Pre-caching the dataset in advance and deserializing it during training proved to be an effective approach. Furthermore, the training process 
demonstrated a degree of increased instability compared to the single-GPU configuration. (probaly b_s) Initially, there was speculation that 
the my ddp training contains bugs. However, after several hours of investigation, it was determined that the issue was related to batch 
size configuration. Specifically, setting #GPU = 1 with a batch size of 64 and #GPU = 8 with a batch size of 8 yielded successful 
results (verified at checkpoint 9). This finding suggests that increasing the batch size does not necessarily lead to better performance. 
Overall, the summary is as follows:

- d_model: 128 is sufficient; larger sizes unnecessary and yielding worse result. This may due to the constrained features (faces only) in the dataset.
- Encoder/Decoder: Scale down accordingly to 128.
- vocab_size: Determines output detail—choose based on desired granularity. 128-256 will be fine.
- width: Should align with vocabulary size (higher resolution supports more detailed outputs).
- Batch size: Must be <=16; otherwise causes severe performance degradation.
- lr: Needs to be relatively large to compensate for small batch size and model capacity. (4e-4 ~ 1e-3).

# GPU=8, micro_batch_size=8, (d_model, vocab_size, width) = (128, 128, 128), trained on 27000 128*128 pictures with 128 epoches, lr=1e-3: loss ~= 0.010
# GPU=8, micro_batch_size=8, (d_model, vocab_size, width) = (128, 256, 256), trained on 27000 256 * 256 pictures with 128 epoches, lr=1e-3: loss ~= 0.013 [*]
# GPU=8, micro_batch_size=8, (d_model, vocab_size, width) = (256, 256, 256), trained on 27000 256 * 256 pictures with 128 epoches, lr=1e-3: loss ~= 0.080
# GPU=8, micro_batch_size=8, (d_model, vocab_size, width) = (256, 256, 256), trained on 27000 256 * 256 pictures with 128 epoches, lr=4e-4: loss ~= 0.080
# GPU=8, micro_batch_size=16, (d_model, vocab_size, width) = (256, 256, 256), trained on 27000 256 * 256 pictures with 128 epoches, lr=4e-4: loss ~= 0.030
# GPU=8, micro_batch_size=32, (d_model, vocab_size, width) = (256, 256, 256), trained on 27000 256 * 256 pictures with 128 epoches, lr=4e-4: loss ~= 0.045
# GPU=8, micro_batch_size=32, (d_model, vocab_size, width) = (256, 512, 128), trained on 190000 128*128 pictures with 64 epoches, bad result loss ~= 0.030
# GPU=8, micro_batch_size=8, (d_model, vocab_size, width) = (512, 512, 128), trained on 190000 128*128 pictures with 100 epoches, bad result loss ~= 0.070



~~~~~~~偏偏念你生生别离~(●'◡'●) 

"""

_dataset_hack = False
_dataset_hack_dupilcation = 3



# um_workers=3, no list etc
def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# AI generated code
class celeba_hqr(Dataset):
    def __init__(self, cache_file_name):
        assert os.path.exists(cache_file_name)
        self.data = torch.load(cache_file_name, weights_only=True)
        self.l = len(self.data)

    def __len__(self):
        if _dataset_hack:
            return self.l * _dataset_hack_dupilcation
        else:
            return self.l
    
    def __getitem__(self, idx):
        if _dataset_hack:
            idx = idx % _dataset_hack_dupilcation
        return self.data[idx] # (C, H, W)
    
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

class block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        tmp = self.relu(x)
        tmp = self.conv1(tmp)
        tmp = self.relu(tmp)
        tmp = self.conv2(tmp)
        return x + tmp

class vqvae(nn.Module):
    def __init__(self, K, d_model, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(input_dim, d_model, 4, 2, 1), nn.ReLU(), 
                                     nn.Conv2d(d_model, d_model, 4, 2, 1), nn.ReLU(), 
                                     nn.Conv2d(d_model, d_model, 3, 1, 1),
                                     block(d_model), block(d_model))
        self.decoder = nn.Sequential(nn.Conv2d(d_model, d_model, 3, 1, 1), # why not relu here
                                     block(d_model), block(d_model),
                                     nn.ConvTranspose2d(d_model, d_model, 4, 2, 1), nn.ReLU(),
                                     nn.ConvTranspose2d(d_model, input_dim, 4, 2, 1))
        self.emd = nn.Embedding(K, d_model)
    
    def forward(self, x: torch.Tensor): # (N, d, H, W)
        z = self.encoder(x).transpose(-1,-3) # (N, d, H, W) -> (N, W, H, d)
        codebook = self.emd.weight.data # (K, d) without 

        # codebook = self.emd.weight can be trained? Seems ok!

        distance = z.unsqueeze(-1).transpose(-1,-2) - codebook # (N, W, H, K, d)
        distance = torch.linalg.vector_norm(distance, dim=-1) # (N, W, H, K)
        codes = torch.argmin(distance, dim=-1) # (N, W, H)

        z_q = self.emd(codes) # (N, W, H, d)

        return z, z_q, self.decoder((z + (z_q - z).detach()).transpose(-1,-3))

def train(rank, world_size, conf):
    # init cluster
    local_rank = rank # single node
    torch.cuda.set_device(local_rank)

    setup(rank, world_size, "nccl")
    if (rank == 0):
        print("Cluster inited.")

    device = torch.device("cuda", local_rank)
    epoch = conf["epoch"]
    beta = conf["beta"]
    gamma = conf["gamma"]
    eval_output_dir = conf["eval_output_dir"]

    d_model = conf["d_model"]
    vocab_size = conf["vocab_size"]
    input_channels = conf["input_channels"]
    batch_size = conf["batch_size"]
    _t = conf["_t"]
    _e = conf["_t"]
    
    ddp_model = DDP(vqvae(vocab_size, d_model, input_channels).to(device), device_ids=[local_rank], output_device=local_rank) #?

    # shit code, each will process, FIX later
    train_dataset = celeba_hqr(_t)
    eval_dataset = celeba_hqr(_e)
    dist.barrier()
 
    if(rank == 0):
        print("Processed all images for all ranks")

    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=3, shuffle=(train_sampler is None))

    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, sampler=eval_sampler, num_workers=3, shuffle=(eval_sampler is None))

    opt = torch.optim.AdamW(ddp_model.parameters(), lr=8e-4) # 1/math.sqrt(8) * 1e-3
    slr = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epoch)
    criterion = nn.MSELoss()

    if(rank == 0):
        print("Entry training loop")
    ddp_model.train()

    for e in range(epoch):
        # train
        train_sampler.set_epoch(e)
        for idx, images in enumerate(train_dataloader):

            input = images.to(device)
            z, z_q, _x = ddp_model(input)
            
            loss = criterion(input, _x) + beta * criterion(z.detach(), z_q) + gamma *  criterion(z, z_q.detach())
            loss.backward() # dist sync here
            opt.step()
            opt.zero_grad()

            dist.reduce(loss, 0)
            if (rank == 0):
                print(f"epoch {e}, step {idx}, avg_loss {loss / world_size}")
        dist.barrier()

        # eval
        with torch.inference_mode():
            if ((e+1) % 16 == 0):
                total_re_loss = torch.tensor(0.0).to(device)
                total_loss = torch.tensor(0.0).to(device)
                num_images = torch.tensor(0).to(device)

                for idx, images in enumerate(eval_dataloader):
                    input = images.to(device)
                    z, z_q, _x = ddp_model(input)

                    batch_size = images.shape[0]
                
                    re_loss = criterion(input, _x) * batch_size
                    loss = re_loss + (beta + gamma) * criterion(z, z_q) * batch_size

                    total_re_loss += re_loss
                    total_loss += loss
                    num_images += batch_size
            
                total_re_loss /= num_images
                total_loss /= num_images
                dist.reduce(total_re_loss, 0)
                dist.reduce(total_loss, 0)
                if rank == 0:
                    print(f"epoch {e}, reconstruction loss {total_re_loss / world_size}, loss {total_loss / world_size}")

        # sampling
        with torch.inference_mode():
            # sampling a batch of images
            if ((e+1) % 16 == 0) and (rank == 0):
                # save check point
                torch.save(ddp_model.module.state_dict(), f"./ckpt/{e}.pt")

                eval_sampler.set_epoch(e)
                # sampling
                for idx, images in enumerate(eval_dataloader):
                    input = images.to(device)
                    _, _, _x = ddp_model(input)

                    for i in range(images.shape[0]):
                        save_tensor_to_image(images[i], eval_output_dir+f"/epoch{e}", f"{i}_E0.png")
                        save_tensor_to_image(_x[i], eval_output_dir+f"/epoch{e}", f"{i}_E1.png")
                    break
        slr.step()
        dist.barrier()

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
    
    print("Stacking cache")
    stacked_tensor = torch.stack(data) 
    print("Writing cache")
    torch.save(stacked_tensor, cache_file_name)
        
if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("Fuck.")
        exit(0)
    world_size = 8
    conf = {
        "epoch" : 192,
        "beta" : 1.0,
        "gamma" : 0.25,
        "eval_output_dir" : "./output",
        "d_model" : 128,
        "batch_size" : 8,
        "vocab_size" : 256,
        "input_channels": 3,
        "width": 256,
        "train_data_path" : "./celeba_hqr/train",
        "eval_data_path" : "./celeba_hqr/eval",
        "_t" : "./t.pt",
        "_e" : "./e.pt",
    }
    # main process handle cache
    print("Preparing data")
    # it turns out that the size of .pt is too large
    init_dataset_cache(conf["train_data_path"], conf["width"], conf["_t"])
    init_dataset_cache(conf["eval_data_path"], conf["width"], conf["_e"])
    print("Launching training mp")
    mp.spawn(fn=train, args=(world_size, conf), nprocs=world_size, join=True) # To be trained on 8*H800
