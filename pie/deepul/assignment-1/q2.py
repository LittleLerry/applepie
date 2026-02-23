import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

import numpy as np
from deepul.hw1_helper import (
    # Q1
    visualize_q1_data,
    q1_sample_data_1,
    q1_sample_data_2,
    q1_save_results,
    # Q2
    q2a_save_results,
    q2b_save_results,
    visualize_q2a_data,
    visualize_q2b_data,
    # Q3
    q3ab_save_results,
    q3c_save_results,
    # Q4
    q4a_save_results,
    q4b_save_results,
    # Q5
    visualize_q5_data,
    q5a_save_results,
    # Q6
    visualize_q6_data,
    q6a_save_results,
)

class masked_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = "zeros", device=None, dtype=None, mask_type = "A"):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        assert kernel_size % 2 == 1
        bound = kernel_size**2 // 2 - 1 if mask_type == "A" else kernel_size**2 // 2
        self.register_buffer("mask", ((torch.arange(kernel_size)[:, None] * kernel_size + torch.arange(kernel_size)[None, :] ) <= bound) * 1.0)

    def forward(self, input):
        self.weight.data *= self.mask
        return super().forward(input)

class residual_block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = masked_conv(in_channels//2, in_channels//2, kernel_size=3, padding=1,mask_type="B")
        self.conv3 = nn.Conv2d(in_channels//2, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        y = self.relu(self.conv1(input))
        y = self.relu(self.conv2(y))
        y = self.relu(self.conv3(y))
        return input + y

class block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = masked_conv(in_channels, in_channels, kernel_size=1, mask_type="B")
    def forward(self, input):
        return self.relu(self.conv1(input))

class pixelCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        # layer 1
        self.conv1 = masked_conv(in_channels, 128, 7, padding=3, mask_type="A")
        # layer 2
        self.residual_blocks = nn.Sequential(*[residual_block(128) for _ in range(8)])
        # layer 3
        self.blocks = nn.Sequential(*[block(128) for _ in range(2)])

        self.out = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, input):
        input = self.relu(self.conv1(input))
        input = self.residual_blocks(input)
        input = self.blocks(input)
        return self.out(input)

class dataset(Dataset):
    def __init__(self, train_data):
        # N H W C -> N C H W
        self.data = torch.from_numpy(train_data).permute(0, 3, 1, 2).long()
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx]

def eval(model, test_data, color_space):
    with torch.inference_mode():
        criterion = nn.CrossEntropyLoss()
        logits = model(test_data.float())
        if color_space == 4:
            loss_r = criterion(logits[:,0:4], test_data[:,0])
            loss_g = criterion(logits[:,4:8], test_data[:,1])
            loss_b = criterion(logits[:,8:], test_data[:,2])
            loss = (loss_r + loss_g + loss_b) / 3
        else:
            loss = criterion(logits[:,0:4], test_data[:,0])

        return loss.cpu()


def q2(train_data, test_data, image_shape, dset_id):

    train_dataset = dataset(train_data)
    train_dataloader = DataLoader(train_dataset, 128)

    device = "cuda:0"
    test_data = torch.from_numpy(test_data).permute(0, 3, 1, 2).long().to(device)
    if len(image_shape) == 3:
        H, W, C = image_shape
        color_space = 4
    else:
        H, W = image_shape
        C = 1
        color_space = 2

    model = pixelCNN(C, color_space * C).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_losses = []
    test_losses = []
    samples = torch.zeros(size=(100, C, H, W)).to(device)

    criterion = nn.CrossEntropyLoss()
    test_losses.append(eval(model, test_data, color_space))
    print("Entry training loop")
    model.train()
    for epoch in range(10):
        for idx, image in enumerate(train_dataloader):

            image = image.to(device)
            logits = model(image.float())
            
            # logits.red.shape = (N, 4, H, W)
            # image.red.shape = (N, H, W)
            if color_space == 4:
                loss_r = criterion(logits[:,0:4], image[:,0])
                loss_g = criterion(logits[:,4:8], image[:,1])
                loss_b = criterion(logits[:,8:], image[:,2])
                loss = (loss_r + loss_g + loss_b) / 3
            else:
                loss = criterion(logits[:,0:4], image[:,0])

            # loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, image)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            print(f"epoch{epoch+1}, step{idx}, loss{loss}")
            train_losses.append(loss.detach().cpu())

            opt.step()
            opt.zero_grad()
        test_losses.append(eval(model, test_data, color_space))
    
    # greedy decoding will not work
    with torch.inference_mode():
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    logits = model(samples)[:,4*c:4*(c+1),h,w] # [B, 4]
                    probs = torch.softmax(logits, dim=-1)
                    samples[:,c,h,w] = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
    return np.array(train_losses), np.array(test_losses), samples.permute(0, 2, 3, 1).cpu().numpy()

q2a_save_results(1, q2)
q2a_save_results(2, q2)

q2b_save_results(1, 'b', q2)
q2b_save_results(2, 'b', q2)