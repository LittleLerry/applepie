import torch
import triton
import triton.language as tl
import torch

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
    ):

    # block definitions
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    allow_tf32: tl.constexpr = True

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
    for _ in range(loop):

        k_j = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero") # no need check for D, shape B_k * D
        v_j = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero") # no need check for D, shape B_k * D

        # FKING idiot side effect, see https://github.com/triton-lang/triton/issues/4574
        # if set allow_tf32 = False, triton impl is MUCH MORE SLOWER THAN torch impl.
        """
        Parameters
        asm        : assembly to run. Must match target’s assembly format.
        constraints: asm constraints in LLVM format. See https://llvm.org/docs/LangRef.html#inline-asm-constraint-string 
        args       : the input tensors, whose values are passed to the asm block
        dtype      : the element type(s) of the returned tensor(s)
        is_pure    : if true, the compiler assumes the asm block has no side-effects
        pack       : the number of elements to be processed by one instance of inline assembly
        """
        # quick transform
        rounded_k_j = tl.inline_asm_elementwise(ASM, "=r, r", [k_j], dtype=tl.float32, is_pure=True, pack=1) if (allow_tf32 and k_j.dtype == tl.float32) else k_j
        rounded_v_j = tl.inline_asm_elementwise(ASM, "=r, r", [v_j], dtype=tl.float32, is_pure=True, pack=1) if (allow_tf32 and v_j.dtype == tl.float32) else v_j
        rounded_q_i = tl.inline_asm_elementwise(ASM, "=r, r", [q_i], dtype=tl.float32, is_pure=True, pack=1) if (allow_tf32 and q_i.dtype == tl.float32) else q_i


        s_i = tl.dot(rounded_q_i, rounded_k_j.trans(), allow_tf32 = allow_tf32) * scale # B_q * B_k
        # tl.device_print("s_i:", s_i)
        
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

def test():
    import random, time, math, tqdm
    random.seed(0)
    torch.set_printoptions(precision=8)

    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    b = 4
    seq = 128 * 128 # * 128
    d_k = 64
    Q_TILE_SIZE = 128
    K_TILE_SIZE = 128
    q = torch.randn(size=(b,seq,d_k), device=DEVICE, dtype=torch.float16).contiguous() 
    k = torch.randn(size=(b,seq,d_k), device=DEVICE, dtype=torch.float16).contiguous() 
    v = torch.randn(size=(b,seq,d_k), device=DEVICE, dtype=torch.float16).contiguous() 
    o = torch.empty(size=(b,seq,d_k), device=DEVICE, dtype=torch.float16).contiguous()
    scale = 1 / math.sqrt(d_k)
    l = torch.empty(size=(b,seq,), dtype=torch.float16, device=DEVICE).contiguous()
    steps = 1
    # --------------------------------------------------------------------------------
    start = time.time()
    for _ in tqdm.tqdm(range(steps),
              desc="Loading",
              total=steps,
              ascii=False,
              miniters=1,
              colour='green',
              dynamic_ncols=True):
        flash_fwd_kernel[(seq // Q_TILE_SIZE, b)](q,k,v,o,l,
                                              q.stride(0),q.stride(1),q.stride(2),
                                              k.stride(0),k.stride(1),k.stride(2),
                                              v.stride(0),v.stride(1),v.stride(2),
                                              o.stride(0),o.stride(1),o.stride(2),
                                              l.stride(0),l.stride(1),
                                              N_QUERIES=seq, N_KEYS=seq, scale=scale, D=d_k, Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE)
        torch.cuda.synchronize()
    end = time.time()
    print(f"triton_impl_t:",end-start)
    # --------------------------------------------------------------------------------
    start = time.time()
    for _ in tqdm.tqdm(range(steps),
              desc="Loading",
              total=steps,
              ascii=False,
              miniters=1,
              colour='green',
              dynamic_ncols=True): 
        S = q @ k.transpose(-2,-1) * scale
        # print(f"S:\n{S}")
        A = S - torch.logsumexp(S,dim=-1,keepdim=True)
        ref_o = (torch.exp(A) @ v).to(torch.float16)
        torch.cuda.synchronize()
    end = time.time()
    print(f"pytorch_impl_t:",end-start)
    # --------------------------------------------------------------------------------
    print(f"o:\n{o.shape}")
    print(f"ref_o:\n{ref_o.shape}")
    print(f"MSE(o, ref_o):{torch.square(o - ref_o).sum()}, SQ(o)={torch.square(o).sum()}, SQ(ref_o)={torch.square(ref_o).sum()}")


def test_timing_flash_forward_backward():
    n_heads =16
    d_head =64
    sequence_length =16384
    q, k, v =torch.randn(
        3, n_heads, sequence_length, d_head,device='cuda', dtype=torch.bfloat16,requires_grad=True
    )

    flash = torch.compile(FlashAttention2.apply)
    def flash_forward_backward():
        o = flash(q, k, v, True)
        loss = o.sum()
        loss.backward()
    
    results = triton.testing.do_bench(flash_forward_backward, rep=10000, warmup=1000)
    print(results)


if __name__ == '__main__':
    test()
