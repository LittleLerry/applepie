import torch
from typing import Type
import triton
import triton.language as tl

"""
`可是我操作的菜，职业不会再回来` --otto

Flashattention Implmentation v2
opt: 
(1) skip masked directly

opt to be done:
(1) TMA
(2) tl.autotune
(3) flash attn backwd
(4) ???
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
    # balanced computation 
    # tl.cdiv(N_QUERIES, Q_TILE_SIZE) must be even!
    _qtotal = tl.cdiv(N_QUERIES, Q_TILE_SIZE) - 1
    q_seq_index_start_1 = query_tile_index * Q_TILE_SIZE
    q_seq_index_start_2 = (_qtotal - query_tile_index) * Q_TILE_SIZE

    Balance_Q_block_ptr_1 = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    Balance_Q_block_ptr_2 = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=((_qtotal - query_tile_index )* Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    Balance_O_block_ptr_1 = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D), 
        strides=(stride_oq, stride_od), 
        offsets=(query_tile_index * Q_TILE_SIZE, 0), 
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    Balance_O_block_ptr_2 = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D), 
        strides=(stride_oq, stride_od), 
        offsets=((_qtotal - query_tile_index) * Q_TILE_SIZE, 0), 
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    Balance_L_block_ptr_1 = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb, 
        shape=(N_QUERIES,), 
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,), 
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    Balance_L_block_ptr_2 = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,), 
        strides=(stride_lq,), 
        offsets=((_qtotal - query_tile_index) * Q_TILE_SIZE,), 
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    # unchanged
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D), 
        strides=(stride_kk, stride_kd), 
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D), # the shape of single batch: seq * d_k, seq = N_QUERIES, d_k = D
        strides=(stride_vk, stride_vd), # define the strides of seq * d_k
        offsets=(0, 0), # within the batch, k starts from (0,0)
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    ) # V_block_ptr = V_block_ptr.advance((K_TILE_SIZE,0))

    loop = tl.cdiv(N_KEYS, K_TILE_SIZE)

    # ===================================================================
    q_i_1 = tl.load(Balance_Q_block_ptr_1, boundary_check=(0,), padding_option="zero") 
    o_i_1 = tl.zeros((Q_TILE_SIZE,D), dtype=tl.float32) # B_q * d
    l_i_1 = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) # B_q
    m_i_1 = tl.full((Q_TILE_SIZE,), -float('inf') , dtype=tl.float32) # B_q

    q_i_2 = tl.load(Balance_Q_block_ptr_2, boundary_check=(0,), padding_option="zero") 
    o_i_2 = tl.zeros((Q_TILE_SIZE,D), dtype=tl.float32) # B_q * d
    l_i_2 = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) # B_q
    m_i_2 = tl.full((Q_TILE_SIZE,), -float('inf') , dtype=tl.float32) # B_q

    part_1_done = False
    part_2_done = False
    assert is_causal == True
    # only for causal, otherwise will crash.
    for idx in range(loop):
        # load once
        k_j = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        v_j = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        if not part_1_done:
            s_i_1 = tl.dot(q_i_1, k_j.trans(), allow_tf32 = allow_tf32) * scale
            if is_causal and ((idx * K_TILE_SIZE + K_TILE_SIZE - 1) > q_seq_index_start_1): # safe: idx * K_TILE_SIZE + K_TILE_SIZE - 1 <= q_seq_index_start_1
                if ((idx * K_TILE_SIZE >= (q_seq_index_start_1 + Q_TILE_SIZE))): # all masked OR reach to the end, for part 1, it is not possibale to reach END with mask!
                    # save result, done
                    part_1_done = True
                    inv_l_i = 1.0 / l_i_1
                    output = inv_l_i[:, None] * o_i_1
                    L_i = m_i_1 + tl.log(l_i_1)
                    tl.store(Balance_O_block_ptr_1, output.to(Balance_O_block_ptr_1.type.element_ty), boundary_check=(0,))
                    tl.store(Balance_L_block_ptr_1, L_i.to(Balance_L_block_ptr_1.type.element_ty), boundary_check=(0,))
                else:
                    i_indices = tl.arange(0, Q_TILE_SIZE)[:, None] + q_seq_index_start_1
                    j_indices = tl.arange(0, K_TILE_SIZE)[None, :] + idx * K_TILE_SIZE
                    s_i_1 = s_i_1 + tl.where(i_indices >= j_indices, 0.0, float('-inf'))

            updated_max = tl.maximum(tl.max(s_i_1, axis = -1), m_i_1)
            p_i = tl.exp(s_i_1 - updated_max[:,None])
            _temp =  tl.exp(m_i_1 - updated_max)
            l_i_1 =  _temp * l_i_1 + tl.sum(p_i, axis = -1)
            o_i_1 = tl.dot(p_i.to(v_j.dtype), v_j, acc = _temp[:,None] * o_i_1, allow_tf32 = allow_tf32)
            m_i_1 = updated_max
        
        if not part_2_done:
            s_i_2 = tl.dot(q_i_2, k_j.trans(), allow_tf32 = allow_tf32) * scale

            if is_causal and ((idx * K_TILE_SIZE + K_TILE_SIZE - 1) > q_seq_index_start_2): # safe: idx * K_TILE_SIZE + K_TILE_SIZE - 1 <= q_seq_index_start_1
                # add mask anyway
                i_indices = tl.arange(0, Q_TILE_SIZE)[:, None] + q_seq_index_start_2
                j_indices = tl.arange(0, K_TILE_SIZE)[None, :] + idx * K_TILE_SIZE
                s_i_2 = s_i_2 + tl.where(i_indices >= j_indices, 0.0, float('-inf'))

                # should we stop or skip? If so, update var and save result
                if ((idx * K_TILE_SIZE >= (q_seq_index_start_2 + Q_TILE_SIZE)) or (idx == loop - 1)):
                    # update
                    updated_max = tl.maximum(tl.max(s_i_2, axis = -1), m_i_2)
                    p_i = tl.exp(s_i_2 - updated_max[:,None])
                    _temp =  tl.exp(m_i_2 - updated_max)
                    l_i_2 =  _temp * l_i_2 + tl.sum(p_i, axis = -1)
                    o_i_2 = tl.dot(p_i.to(v_j.dtype), v_j, acc = _temp[:,None] * o_i_2, allow_tf32 = allow_tf32)
                    m_i_2 = updated_max
                    # save 
                    part_2_done = True
                    inv_l_i = 1.0 / l_i_2
                    output = inv_l_i[:, None] * o_i_2
                    L_i = m_i_2 + tl.log(l_i_2)
                    tl.store(Balance_O_block_ptr_2, output.to(Balance_O_block_ptr_2.type.element_ty), boundary_check=(0,))
                    tl.store(Balance_L_block_ptr_2, L_i.to(Balance_L_block_ptr_2.type.element_ty), boundary_check=(0,))

            # if not done, update them. How ever, we do not use condition to reduce warp div.
            updated_max = tl.maximum(tl.max(s_i_2, axis = -1), m_i_2)
            p_i = tl.exp(s_i_2 - updated_max[:,None])
            _temp =  tl.exp(m_i_2 - updated_max)
            l_i_2 =  _temp * l_i_2 + tl.sum(p_i, axis = -1)
            o_i_2 = tl.dot(p_i.to(v_j.dtype), v_j, acc = _temp[:,None] * o_i_2, allow_tf32 = allow_tf32)
            m_i_2 = updated_max
            


        # unsupported AST node type: Break
        # See: https://github.com/triton-lang/triton/issues/3157 
        # if part_1_done and part_2_done:
        #    break

        # move to next block
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE,0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE,0))


def test():
    import random, math
    random.seed(0)
    torch.set_printoptions(precision=8)

    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    b = 16
    seq = 128
    d_k = 128
    Q_TILE_SIZE = 16
    K_TILE_SIZE = 16
    q = torch.randn(size=(b,seq,d_k), device=DEVICE, dtype=torch.float32).contiguous() 
    k = torch.randn(size=(b,seq,d_k), device=DEVICE, dtype=torch.float32).contiguous() 
    v = torch.randn(size=(b,seq,d_k), device=DEVICE, dtype=torch.float32).contiguous() 
    o = torch.empty(size=(b,seq,d_k), device=DEVICE, dtype=torch.float32).contiguous()
    scale = 1 / math.sqrt(d_k)
    l = torch.empty(size=(b,seq,), dtype=torch.float32, device=DEVICE).contiguous()

    S = q @ k.transpose(-1,-2) * scale
    S = torch.where(torch.arange(0, seq, device=DEVICE)[:,None] < torch.arange(0, seq, device=DEVICE)[None, :], -1e6, S)
    ref_o = torch.softmax(S ,dim=-1) @ v
    _0_threads = (seq - 1) // Q_TILE_SIZE + 1
    # --------------------------------------------------------------------------------
    flash_fwd_kernel[(_0_threads // 2, b)](q,k,v,o,l,
                                              q.stride(0),q.stride(1),q.stride(2),
                                              k.stride(0),k.stride(1),k.stride(2),
                                              v.stride(0),v.stride(1),v.stride(2),
                                              o.stride(0),o.stride(1),o.stride(2),
                                              l.stride(0),l.stride(1),
                                              N_QUERIES=seq, N_KEYS=seq, scale=scale, D=d_k, Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE, is_causal = True, allow_tf32=False)
    print(f"{(ref_o - o).abs().sum()}")

if __name__ == '__main__':
    test()



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
        _0_threads = (seq - 1) // Q_TILE_SIZE + 1
        assert _0_threads % 2 == 0

        flash_fwd_kernel[(_0_threads // 2, batch_size)](Q, K, V, O, L, 
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
            S =torch.where(torch.arange(n_queries, device=S.device)[None, :, None] >= torch.arange(n_keys, device=S.device)[None, None, :], S, float('-inf'))

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