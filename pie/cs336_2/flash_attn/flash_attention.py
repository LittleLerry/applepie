import torch
from typing import Type
import triton
import triton.language as tl

"""
From now, I believe my life contains only failures and errors.
I even cannot debug it, cause the `os` rejects transfer its control to my hand.

Native Flashattention Implmentation v1
"""
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    allow_tf32: tl.constexpr,
    ):

    # block definitions
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    q_seq_index_start = query_tile_index * Q_TILE_SIZE

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D), # the shape of single batch: seq * d_k, seq = N_QUERIES, d_k = D
        strides=(stride_qq, stride_qd), # define the strides of seq * d_k
        offsets=(query_tile_index * Q_TILE_SIZE, 0), # within the batch, offset(element, element)
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D), # the shape of single batch: seq * d_k, seq = N_QUERIES, d_k = D
        strides=(stride_oq, stride_od), # define the strides of seq * d_k
        offsets=(query_tile_index * Q_TILE_SIZE, 0), # within the batch, offset(element, element)
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb, # >>>>>>>>>>>>>>>>>>
        shape=(N_QUERIES,), # the shape of single batch: seq
        strides=(stride_lq,), # define the strides of seq
        offsets=(query_tile_index * Q_TILE_SIZE,), # within the batch, offset(element,)
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D), # the shape of single batch: seq * d_k, seq = N_QUERIES, d_k = D
        strides=(stride_kk, stride_kd), # define the strides of seq * d_k
        offsets=(0, 0), # within the batch, k starts from (0,0)
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    ) # K_block_ptr = K_block_ptr.advance((K_TILE_SIZE,0))

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D), # the shape of single batch: seq * d_k, seq = N_QUERIES, d_k = D
        strides=(stride_vk, stride_vd), # define the strides of seq * d_k
        offsets=(0, 0), # within the batch, k starts from (0,0)
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    ) # V_block_ptr = V_block_ptr.advance((K_TILE_SIZE,0))

    # var definitions
    loop = tl.cdiv(N_KEYS, K_TILE_SIZE)
    q_i = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero") # no need check for D, shape B_q * D 
    # The on chip buffers should have dtype tl.float32. 
    o_i = tl.zeros((Q_TILE_SIZE,D), dtype=tl.float32) # B_q * d
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) # B_q
    m_i = tl.full((Q_TILE_SIZE,), -float('inf') , dtype=tl.float32) # B_q
    # ------------------------------------------------------
    # There are fucking idiot accurate errors here
    # See: https://github.com/triton-lang/triton/issues/4574
    ASM: tl.constexpr = "cvt.rna.tf32.f32 $0, $1;" 
    # ------------------------------------------------------
    assert Q_block_ptr.type.element_ty == K_block_ptr.type.element_ty and Q_block_ptr.type.element_ty == V_block_ptr.type.element_ty
    # flash attn fused kernel
    for idx in range(loop):

        k_j = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero") # no need check for D, shape B_k * D
        v_j = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero") # no need check for D, shape B_k * D


        rounded_k_j = tl.inline_asm_elementwise(ASM, "=r, r", [k_j], dtype=tl.float32, is_pure=True, pack=1) if (allow_tf32 and k_j.dtype == tl.float32) else k_j
        rounded_v_j = tl.inline_asm_elementwise(ASM, "=r, r", [v_j], dtype=tl.float32, is_pure=True, pack=1) if (allow_tf32 and v_j.dtype == tl.float32) else v_j
        rounded_q_i = tl.inline_asm_elementwise(ASM, "=r, r", [q_i], dtype=tl.float32, is_pure=True, pack=1) if (allow_tf32 and q_i.dtype == tl.float32) else q_i

        s_i = tl.dot(rounded_q_i, rounded_k_j.trans(), allow_tf32 = allow_tf32) * scale # B_q * B_k

        # causul mask
        if is_causal:
            i_indices = tl.arange(0, Q_TILE_SIZE)[:, None] + q_seq_index_start
            j_indices = tl.arange(0, K_TILE_SIZE)[None, :] + idx * K_TILE_SIZE
            mask = tl.where(i_indices >= j_indices, 0.0, -1e6)
            s_i = s_i + mask
        
        updated_max = tl.maximum(tl.max(s_i, axis = -1), m_i) # B_q
        p_i = tl.exp(s_i - updated_max[:,None]) # B_q * B_k

        # update l_i and o_i 
        _temp =  tl.exp(m_i - updated_max)
        l_i =  _temp * l_i + tl.sum(p_i, axis = -1)
        rounded_p_i = tl.inline_asm_elementwise(ASM, "=r, r", [p_i], dtype=tl.float32, is_pure=True, pack=1) if (allow_tf32 and p_i.dtype == tl.float32) else p_i
        o_i = tl.dot(rounded_p_i.to(rounded_v_j.dtype), rounded_v_j, acc = _temp[:,None] * o_i, allow_tf32 = allow_tf32)

        # update max
        m_i = updated_max

        # move to next k,v block
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE,0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE,0))

    inv_l_i = 1.0 / l_i

    output = inv_l_i[:, None] * o_i
    L_i = m_i + tl.log(l_i)

    tl.store(O_block_ptr, output.to(O_block_ptr.type.element_ty), boundary_check=(0,)) # no need check for D
    tl.store(L_block_ptr, L_i.to(L_block_ptr.type.element_ty), boundary_check=(0,))


class flashattnTriton(torch.autograd.Function):
    # batch_size, sequence_length,d_head,
    @staticmethod
    def forward(ctx, Q, K, V, is_causal):
        d_k = Q.shape[-1]
        seq = Q.shape[-2]
        orignal_shape = Q.shape[:-2]

        Q = Q.view(-1,seq, d_k).contiguous()
        V = V.view(-1,seq, d_k).contiguous()
        K = K.view(-1,seq, d_k).contiguous()
        batch_size = Q.shape[0]
        O = torch.empty(size=(batch_size,seq,d_k,), device=Q.device)
        L = torch.empty(size=(batch_size,seq, ), device=Q.device)
        scale = 1 / (d_k ** 0.5)

        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        assert (d_k % 16 == 0) and (Q.shape[1] % 16 == 0) and (V.shape[1] == Q.shape[1]) and (K.shape[1] == Q.shape[1])

        flash_fwd_kernel[((seq - 1) // Q_TILE_SIZE + 1, batch_size)](Q, K, V, O, L, 
                        Q.stride(0),Q.stride(1),Q.stride(2),
                        K.stride(0),K.stride(1),K.stride(2),
                        V.stride(0),V.stride(1),V.stride(2),
                        O.stride(0),O.stride(1),O.stride(2),
                        L.stride(0),L.stride(1),
                        N_QUERIES=seq, N_KEYS=seq, scale=scale, D=d_k, 
                        Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE, 
                        is_causal=is_causal, allow_tf32=True)
        
        O = O.view(*orignal_shape, seq, d_k)
        ctx.save_for_backward(Q,K,L,V,O)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, grad_o):
        # Q,K,V,O,dO,L
        # return dQ, dK, dV
        # backward with recomputation
        Q,K,L,V,O = ctx.saved_tensors
        D = (O * grad_o).sum(dim=-1)

        scale = 1 / (Q.shape[-1] ** 0.5)
        S = Q @ K.transpose(-1,-2) * scale
        if ctx.is_causal:
            n_queries = Q.shape[-2]
            n_keys = K.shape[-2]
            S =torch.where(torch.arange(n_queries, device=S.device)[None, :, None] >= torch.arange(n_keys, device=S.device)[None, None, :], S, -1e6)

        P = torch.exp(S - L.unsqueeze(-1))
        dV = P.transpose(-1,-2) @ grad_o
        dP = grad_o @ V.transpose(-1,-2)
        dS = P * (dP - D.unsqueeze(-1))
        dQ = dS @ K * scale
        dK = dS.transpose(-1,-2) @ Q * scale
        # output = forward(input)
        # Accepts:  = dL / doutput
        # Return:   = dL / dinput
        return (dQ, dK, dV, None)






