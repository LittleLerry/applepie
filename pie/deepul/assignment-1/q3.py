"""
build the code, make it works, otherwise you are missing knowledge
PE is stupid thing when doing KV cache, typically, you need manully set offset!

It has been observed that the implementation of KV caching skips masked tokens, which 
theoretically yields a twofold speedup compared to the original implementation. 
However, experimental results have shown that the actual speedup is less than twofold. 
Further investigation is needed to resolve this discrepancy.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import time

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


class FNN(nn.Module):
    def __init__(self, d_model, d_ff, do = 0.1) :
        super().__init__()
        # self.relu = nn.ReLU(inplace=False) # Not sure if ==True works
        self.gelu = nn.GELU()
        self.mff = nn.Linear(d_model, d_ff)
        self.ffm = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.mff(x)
        x = self.gelu(x)
        return self.ffm(x)

class block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, KVCache, max_seq_len, do=0.1):
        super().__init__()
        self.attn = sublayer(MHA(d_model, num_heads, KVCache, max_seq_len), d_model)
        self.fnn = sublayer(FNN(d_model, d_ff), d_model)
    
    def forward(self,x):
        x = self.attn(x)
        return self.fnn(x)
    
    def clear_kvcache(self):
        self.attn._layer.clear()


class sublayer(nn.Module):
    def __init__(self, layer, d_model):
        super().__init__()
        self.layer = layer
        self.ln = nn.LayerNorm(d_model)
    
    def forward(self, x): # pre-Norm residual
        return x + self.layer(self.ln(x))
    
    @property
    def _layer(self):
        return self.layer

# casual
class MHA(nn.Module):
    def __init__(self, d_model, num_heads, KVCache, max_seq_len, do=0.1):
        super().__init__()
        # assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self._toq = nn.Linear(d_model, d_model)
        self._tok = nn.Linear(d_model, d_model)
        self._tov = nn.Linear(d_model, d_model)
        self._o = nn.Linear(d_model, d_model)
        self.max_seq_len = max_seq_len
        # kv cache
        self.kvcache = KVCache
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)

        # full causal mask
        full_causal_mask = (torch.arange(max_seq_len)[:, None] >= torch.arange(max_seq_len)[None, :])
        self.register_buffer("mask",  full_causal_mask, persistent=False)
    
    def forward(self, x):
        # x with shape (, current_seq_len, d_model)
        current_seq_len = x.shape[-2]
        previous_seq_len = 0 if self.k_cache is None else self.k_cache.shape[-2]
        _kv_size = current_seq_len + previous_seq_len

        _q = self._toq(x) # There is no need to save _q
        # print(_q[0,-1])
        _k = self._tok(x)
        _v = self._tov(x)

        shape = _q.shape[:-1]
        scale = math.sqrt(self.d_k) / self.d_k

        _q = _q.view(*shape, self.num_heads, -1).transpose(-2,-3) # [,num_heads,seq,d_k] or [,num_heads,1,d_k]
        _k = _k.view(*shape, self.num_heads, -1).transpose(-2,-3)
        _v = _v.view(*shape, self.num_heads, -1).transpose(-2,-3)

        # restore KV from cache
        if (self.kvcache and torch.is_inference_mode_enabled()):
            if self.k_cache is None:
                self.k_cache, self.v_cache = _k, _v # [...,num_heads,1,d_k]
            else:
                self.k_cache = torch.cat((self.k_cache, _k), dim=-2)
                self.v_cache = torch.cat((self.v_cache, _v), dim=-2) # type: ignore
            _k, _v = self.k_cache, self.v_cache
        # mask

        if (self.kvcache and torch.is_inference_mode_enabled()):
            mask = self.mask[previous_seq_len:_kv_size,:_kv_size] # type: ignore
        else:
            mask = self.mask[:current_seq_len,:current_seq_len] # type: ignore
        
        # For kv cache inference, attn.shape = (, num_heads, current_seq_len, d_k) @ (, num_heads, d_k, _kv_size) = (, num_heads, current_seq_len, _kv_size)
        # then mask is proper size and attn_score.shape = (, num_heads, current_seq_len, _kv_size)
        # then V.shape = (, num_heads, current_seq_len, _kv_size) @ (, num_heads, _kv_size, d_k) = (, num_heads, current_seq_len, d_k)
        # then return shape = (, current_seq_len, d_model) == x.shape
        attn = _q @ _k.transpose(-1,-2) * scale
        # print(f"({mask.shape},{attn.shape})")
        attn_score = torch.softmax(torch.where(mask, attn, float('-inf')), dim=-1)
        V = attn_score @ _v
        O = V.transpose(-2, -3).contiguous().view(*shape, -1)
        return self._o(O)
    
    def clear(self):
        self.k_cache, self.v_cache = None, None
    
# FUCKINGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG bugs here
# If you try to do KV cache, you MUST CONTROL the PE cause each time it starts from different seq index
class pre(nn.Module):
    def __init__(self, n, d_model, max_seq_len):
        super().__init__()
        # print(f"({n},{d_model})")
        self.emd = nn.Embedding(n, d_model)
        self.max_seq_len = max_seq_len
        # [seq, d_model]
        i = torch.arange(0, max_seq_len)[:, None]
        j = torch.arange(0, d_model)[None, :]
        pe = i / ( torch.pow(10000, (j // 2 * 2) / d_model))
        pe = torch.where(j%2==0, torch.sin(pe), torch.cos(pe))
        self.register_buffer("pe", pe)

    def forward(self, x, offset):
        x = self.emd(x)
        # positional encoding
        seq_len = x.shape[-2]
        assert seq_len <= self.max_seq_len
        x = x + self.pe[offset:seq_len+offset,:] # type: ignore
        return x

class post(nn.Module):
    def __init__(self, d_model, d_v):
        super().__init__()
        self.l = nn.Linear(d_model, d_v)
    def forward(self, x):
        return self.l(x)

class TransformerCausalLM(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, num_blocks, max_seq_len, d_v, KVCache = False):
        super().__init__()
        self.pre_layer = pre(d_v, d_model,max_seq_len)
        self.post_layer = post(d_model, d_v)
        self.blocks = nn.Sequential(*[block(d_model, num_heads, d_ff, KVCache, max_seq_len) for _ in range(num_blocks)])
        self.kvcache_enabled = KVCache
    def forward(self, x, offset):
        x = self.pre_layer(x, offset) # Fuck! Must be offset!
        x = self.blocks(x)
        x = self.post_layer(x)
        return x
    
    def clear_kvcache(self):
        for b in self.blocks:
            b.clear_kvcache() # type: ignore
    
    def is_kv_cache_enabled(self):
        return self.kvcache_enabled


# AI generated function
def concat_uint2_to_int_bitwise(tensor):

    N, Seq, C = tensor.shape

    result = torch.zeros(N, Seq, dtype=torch.int64)

    for c in range(C):
        shift = c * 2
        d = tensor[:, :, c].to(torch.int64) # d_i < 4
        result |= d << shift
    
    return result.contiguous()
# AI generated function
def unpack_int_to_uint2_vectorized(packed_tensor, C):

    N, Seq = packed_tensor.shape

    shifts = torch.arange(C) * 2
    masks = ((1 << 2) - 1) << shifts  # (C,), 0b11, 0b1100, 0b110000
    
    packed_expanded = packed_tensor.unsqueeze(-1) # (N, seq, 1)

    unpacked: torch.Tensor = (packed_expanded & masks) >> shifts.view(1, 1, C)
    
    return unpacked.contiguous()

class imageTokenizerDataset(Dataset):
    # N H W C is image input, for each pixel in the image, convert its value to a single token
    # There are C channels, yielding a token with C* `bits_for_value_[n,h,w,c]` bits.
    # The return type should be [Batch_size, seq_len]

    def __init__(self, train_data, sep_token_id):
        data = torch.from_numpy(train_data)
        N, H, W, C = data.shape
        # N H W C -> N H*W
        self.data = concat_uint2_to_int_bitwise(data.view(N,-1,C).contiguous()) # (N, H*W)
        # sep token will be used for both input_ids and label
        self.sep_token_id = torch.tensor([sep_token_id])
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return {
            "input_ids" : torch.cat((self.sep_token_id, self.data[idx])),
            "label" : torch.cat((self.data[idx], self.sep_token_id)),
        }

def q3(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    image_shape: (H, W, C), height, width, and # of channels of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
           used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, C) of samples with values in {0, 1, 2, 3}
    """
    device = "cuda:0"
    # device = "cpu"
    H, W, C = image_shape
    channel_bit = 2

    channel_range = 2**channel_bit
    composed_channel_bit = channel_bit * C
    composed_channel_range = 2 ** composed_channel_bit
    sep_token_id = composed_channel_range
    vocab_size = composed_channel_range + 1
    # train dataset
    train_dataset = imageTokenizerDataset(train_data, sep_token_id)
    train_dataloader = DataLoader(train_dataset, 32)
    # test dataset
    test_dataset = imageTokenizerDataset(test_data, sep_token_id)
    test_dataloader = DataLoader(test_dataset, 32)
    # model confs
    criterion = nn.CrossEntropyLoss()
    model = TransformerCausalLM(128, 256, 4,2 , H*W+1, vocab_size, KVCache=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr = 1e-3)

    train_losses = []
    test_losses = []

    model.train()
    print("Entry training loop.")
    for epoch in range(16):
        # train
        for idx, image in enumerate(train_dataloader):
            input = image["input_ids"].to(device) # [b, seq]
            label = image["label"].to(device) # [b, seq]

            logits = model(input, 0) # [b, seq, vocab_size]
            loss = criterion(logits.view(-1, logits.size(-1)), label.view(-1))
            train_losses.append(loss.detach().cpu())
            print(f"Train/epoch {epoch+1}, step {idx+1}, loss {train_losses[-1]}")

            loss.backward()

            opt.step()
            opt.zero_grad()
        # validate
        with torch.inference_mode():
            total_loss = 0.0
            weights = 0
            for idx, image in enumerate(test_dataloader):
                input = image["input_ids"].to(device)
                label = image["label"].to(device)
                batch_size = input.shape[0]

                logits = model(input, 0)
                loss = criterion(logits.view(-1, logits.size(-1)), label.view(-1)).cpu()
                
                total_loss += loss * batch_size
                weights += batch_size
                # clear cache
                if (model.is_kv_cache_enabled()):
                    model.clear_kvcache()
            test_losses.append((total_loss / weights))
            print(f"Test/epoch {epoch+1}, step {idx+1}, loss {test_losses[-1]}")
        # schuduler
        # pass
    
    # pratical sampling without kv cache
    samples = torch.tensor([sep_token_id]).expand(size=(100, 1)).to(device)
    with torch.inference_mode():
        
        # sampling 
        start = time.time()
        for i in range(H*W): # process samples [B,i+1]
            logits = model(samples[:,-1:], i)[:,-1] if model.is_kv_cache_enabled() else model(samples, 0)[:,-1] # FUCK!!!!!!!OFFSET!!!!
            probs = torch.softmax(logits, dim=-1) # 
            rollout = torch.multinomial(probs[:,:-1], num_samples=1) # [B,1], last vocab is sep token, remove it via slicing
            samples = torch.cat((samples, rollout), dim=-1) # shape [B, i+1]
            #?
        if (model.is_kv_cache_enabled()):
            model.clear_kvcache()
        torch.cuda.synchronize()
        end = time.time()
        print(f"kvcache {model.is_kv_cache_enabled()}, d {end - start}")
        

    samples = unpack_int_to_uint2_vectorized(samples[:,1:].cpu(), C) # [B, H*W] -> [B, H*W, C]

    return np.array(train_losses), np.array(test_losses), samples.view(-1,H,W,C).numpy()


time_list_no_cache, time_list_with_cache, samples_no_cache, samples_with_cache = None, None, None, None

def q3_c(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    image_shape: (H, W, C), height, width, and # of channels of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# sampling steps,) numpy array of time per sampling iteration, without caching
    - a (# sampling steps,) numpy array of time per sampling iteration, with caching
    - a numpy array of size (100, H, C, W) of samples with values in {0, 1, 2, 3} (sample generated without caching)
    - a numpy array of size (100, H, C, W) of samples with values in {0, 1, 2, 3} (sample generated with caching)
    """
    return time_list_no_cache, time_list_with_cache, samples_no_cache, samples_with_cache


q3ab_save_results(1, 'b', q3)
# ===========================================================================
# kvcache True, d 0.4838523864746094
# kvcache False, d 0.7383875846862793
# ===========================================================================