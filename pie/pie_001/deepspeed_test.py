import segmentation_models_pytorch as smp
import torch
import argparse
import os
import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import transforms
from deepspeed.accelerator import get_accelerator

def test(args, engine, optimizer, trainloader):
    pass
        
def infer(args, engine, optimizer, trainloader):
    pass

def train(args, engine, optimizer, trainloader):
    #! deepspeed: get local device again
    local_device = get_accelerator().device_name()
    local_rank = engine.local_rank
    # constants
    betas = torch.linspace(args.beta_start, args.beta_end, args.steps,device=local_device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    # \sqrt(alphas_bar) | \sqrt(1-alphas_bar)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    # loss function
    criterion = nn.MSELoss()
    for epoch in range(args.epochs):
        # run epoch
        running_loss = 0.0
        engine.train()
        for i, (data, _ ) in enumerate(trainloader):
            #! data [N, C, H, W]
            # x_0 ~ q(x_0)
            x_0 = data.to(local_device)
            n, c, h, w = x_0.shape
            # t ~ U({0,...,T-1})
            t = torch.randint(low=0, high=args.steps, size=(n,),device=local_device)
            # e ~ N(0,I)
            noise = torch.randn_like(x_0,device=local_device)
            # e_\theta(.., t)
            input = sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1) * x_0 + sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1) * noise
            noise_prediciton = engine(input, t)
            # |e - e_prediciton|
            loss = criterion(noise_prediciton, noise) / n # avg per_batch loss
            # deepspeed here
            engine.backward(loss)
            engine.step()
            # log, it should be synced...
            running_loss += loss.item()
            if local_rank == 0 and i % args.log_interval == (args.log_interval - 1):
                print(f"[{epoch + 1 : d}, {i + 1 : 5d}] loss: {running_loss / args.log_interval : .8f}")
                running_loss = 0.0
        #! deepspeed: all ranks should call this method to save the model except for rank 0 
        if (epoch % args.save_interval == (args.save_interval - 1)):
            engine.save_checkpoint(save_dir = args.save_dir)
            test(args=args, engine=engine, optimizer=optimizer, trainloader=trainloader)

    print("Training done.")

# Net definition should not contains any device related controlling code
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self._net = smp.Unet("resnet34",  # 224×224x3
                             encoder_depth=4,
                             encoder_weights=None, 
                             decoder_use_norm='batchnorm',
                             decoder_channels=(512, 256, 128, 64),
                             decoder_attention_type='scse',
                             decoder_interpolation='nearest',
                             in_channels=args.channels + 1, # time embedding
                             classes=args.channels,
                             activation=None # should not use activation
                             )
        self._step_embd_1 = nn.Embedding(args.steps, args.t_embedding_dim)
        self.projection = nn.Sequential(
            nn.Linear(args.t_embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, args.hw * args.hw * 1),
            nn.Tanh()  # Normalize to [-1, 1]????? SHOULD WE DO THAT??????
        )
        self._hw = args.hw

    def forward(self, x, t):
        # x = [n, c, h, w]
        # t = [n]
        tm = self.projection(self._step_embd_1(t)).view(-1, 1, self._hw, self._hw)
        return self._net(torch.cat([x, tm], dim=1))

def get_ds_config(args):
    # some dict confs
    #! control training related parameters of models here
    #! opt, scheduler can be managed here
    #! check https://www.deepspeed.ai/docs/config-json/ for more detials
    ds_config = {
        # It is effective training batch size. This is the amount of data samples that leads to one step of model update.
        # train_batch_size must be equal to train_micro_batch_size_per_gpu * gradient_accumulation_steps * number of GPUs
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": 1,
        "train_micro_batch_size_per_gpu": args.batch_size//args.gpus, # GPUs must be 8 in this case
        "steps_per_print": 2000, # ??????????????
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
                "warmup_num_steps": 3000,
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


def add_argument():
    parser = argparse.ArgumentParser(description='deepspeed_test')
    parser.add_argument(
        '--input', default='./ds', type=str, help='input dataset'
    )
    parser.add_argument(
        '--batch_size', default=128, type=int, help='batch size'
    )
    parser.add_argument(
        '--epochs', default=512, type=int, help='# of epochs'
    )
    parser.add_argument(
        '--steps', default=1000, type=int, help='T value'
    )
    parser.add_argument(
        '--beta_start', default=0.0001, type=float, help='b_start for betas'
    )
    parser.add_argument(
        '--beta_end', default=0.02, type=float, help='b_end for betas'
    )
    parser.add_argument(
        '--log_interval', default=50, type=int, help='log interval'
    )
    parser.add_argument(
        '--save_interval', default=40, type=int, help='save epoch interval'
    )
    parser.add_argument(
        '--save_dir', default='./checkpoints', type=str, help='checkpoint will be saved to this dir'
    )
    parser.add_argument(
        '--hw', default=224, type=int, help='h/w value of input images'
    )
    parser.add_argument(
        '--channels', default=3, type=int, help='input channels of input images'
    )
    parser.add_argument(
        '--dtype', default='bf16', type=str, help='training in bf16 or fp16'
    )
    parser.add_argument(
        "--stage",
        default=3,
        type=int,
        choices=[0, 1, 2, 3],
        help="ZeRO stage",
    )
    parser.add_argument(
        '--gpus',
        default=2,
        type=int,
        choices=[1, 2, 4, 8],
        help="# of gpus used",
    )
    #! must have this:
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher", #! deepspeed will do it for us
    )
    parser.add_argument(
        "--t_embedding_dim",
        type=int,
        default=512,
        help="t_embedding_dim",
    )
    # add deepspeed related args
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def main(args):
    # control GPUs used
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args.gpus)))
    # then init
    if(args.local_rank == 0):
        print('Initializing the cluster')

    deepspeed.init_distributed()
    _local_rank = int(os.environ.get('LOCAL_RANK'))
    # eq to torch.cuda.set_device()
    get_accelerator().set_device(_local_rank)

    # prepare the dataset
    transform = transforms.Compose([
        transforms.Resize((args.hw, args.hw)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    dataset = datasets.ImageFolder(root=args.input, transform=transform)
    # num_works should be an constant, the total # of workers will be GPUs * num_workers
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers = 3)
    if(args.local_rank == 0):
        print('Initializing the models')
    model = Net(args=args)
    ds_config = get_ds_config(args)
    #! why we cannot conf dataloader
    # obtain the distribution versions of engine, opt, dataloader
    # engine has been automatically moved to some accs
    engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
        config=ds_config,
    )
    #! _local_rank = int(os.environ.get('LOCAL_RANK'))???????/
    # local_device = get_accelerator().device_name(engine.local_rank)
    local_rank = engine.local_rank
    #! ...
    assert local_rank == _local_rank
    # train funciton
    if(args.local_rank == 0):
        print('Entrying the training function')
    train(args=args,engine=engine,optimizer=optimizer,trainloader=trainloader)

if __name__ == "__main__":
    args = add_argument()
    main(args)