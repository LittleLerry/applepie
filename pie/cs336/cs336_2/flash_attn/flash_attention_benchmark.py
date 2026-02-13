import torch
from flash_attention import flashattnTriton
from native_attention import flashattnTorch
import triton
import itertools
import pandas as pd

def flash_forward_backward(kernel, should_backward ,*data):
    o = kernel(*data)
    if should_backward:
        loss = o.sum()
        loss.backward()

def benchmark():
    batch_size = 1
    seq_len = [2**x for x in range(7,17)]
    d_k = [2**x for x in range(4,8)]
    # kernel = [flashattnTorch.apply, torch.compile(flashattnTorch.apply), flashattnTriton.apply, torch.compile(flashattnTriton.apply)]
    kernel = [flashattnTorch.apply, flashattnTriton.apply]
    # acc = [torch.float16, torch.float32]
    acc = [torch.float16, torch.float32]
    backwards = [False]

    data = []
    print("Entry benchmark.")
    for k, a  in itertools.product(kernel, acc):
        for s, d in itertools.product(seq_len, d_k):
            for bkwd in backwards:
                _q = torch.randn(batch_size, s, d, device='cuda', dtype=a, requires_grad=True)
                _k = torch.randn(batch_size, s, d, device='cuda', dtype=a, requires_grad=True)
                _v = torch.randn(batch_size, s, d, device='cuda', dtype=a, requires_grad=True)
                results = triton.testing.do_bench(lambda: flash_forward_backward(k, bkwd, _q,_k,_v, True), rep=10000, warmup=1000)
                print(f"kernel: {k}, acc: {a}, seq_len: {s}, d_k: {d}, result: {results}")
                data.append({
                    'kernel': k,
                    'dtype': a,
                    'seq_len': s,
                    'd_k': d,
                    'result': results
                })
    
    df = pd.DataFrame(data)
    latex_table = df.to_latex(
        index=False,
        caption="Performance Results",
        label="tab:results",
    )
    # print(latex_table)
    df.to_latex('results.tex', index=False, caption="Performance Results")

if __name__ == '__main__':
    benchmark()