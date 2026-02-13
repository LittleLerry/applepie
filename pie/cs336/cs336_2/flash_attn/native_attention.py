import torch
from typing import Type
"""
Today my friend gived me an orange. everyone likes orange.

Native torchattention Implmentation v1
"""

class flashattnTorch(torch.autograd.Function):
    @staticmethod
    # [batch_size, seq, d_k]
    # [batch_size, num_heads, seq, d_k]
    def forward(ctx, Q, K, V, is_causal):
        d_k = Q.shape[-1]
        scale = 1 / (d_k ** 0.5)
        S = Q @ K.transpose(-1,-2) * scale
        if is_causal:
            n_queries = Q.shape[-2]
            n_keys = K.shape[-2]
            S =torch.where(torch.arange(n_queries, device=S.device)[:, None] >= torch.arange(n_keys, device=S.device)[None,:], S, float('-inf'))
        
        P = S.softmax(dim=-1)
        O = P @ V
        # L = torch.logsumexp(S,dim=-1) # for verify the correctness of the program only

        ctx.save_for_backward(Q,K,V,P) # ctx.save_for_backward(Q,K,V,P,L)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, grad_o):
        # [batch_size, seq, d_k]
        # [batch_size, num_heads, seq, d_k]
        Q,K,V,P = ctx.saved_tensors
        scale = 1 / (Q.shape[-1] ** 0.5)

        dV = P.transpose(-1,-2) @ grad_o
        dP = grad_o @ V.transpose(-1,-2)
        dS = P * dP - P * (P * dP).sum(dim=-1, keepdim=True)
        dQ = dS @ K * scale
        dK = dS.transpose(-1,-2) @ Q * scale
        return (dQ, dK, dV, None)






