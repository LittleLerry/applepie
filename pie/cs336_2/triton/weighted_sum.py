import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
def weighted_sum(x, weight):
    return (weight * x).sum(axis=-1)

"""
Defines a Triton kernel for computing weighted sums with tiled memory access.

This kernel implements a parallel weighted sum operation using Triton's JIT compilation.
The computation is distributed across multiple program instances, each processing 
specific tiles of the input matrices to optimize memory access patterns.

Parameters:
    x_ptr: Pointer to input matrix X (row-major)
    w_ptr: Pointer to weight matrix W (row-major)  
    output_ptr: Pointer to output matrix (row-major)
    rows: Total rows in input matrix
    row_num_tiles: Number of row tiles (rows MAY BE samller than row_num_tiles * row_size_tiles)
    row_size_tiles: Tile size for rows (MUST BE compile-time constant)
    cols: Total columns in input matrix
    col_num_tiles: Number of column tiles (cols MAY BE samller than col_num_tiles * col_size_tiles)
    col_size_tiles: Tile size for columns (MUST BE compile-time constant)

Kernel Launch Strategy:
    - 1D kernel launch where each program instance computes a specific tile
    - User defines which data tile each instance processes, where instance processes can be distinguished by tl.program_id(axis=0)
    - Triton runtime only assigns unique program_id to each parallel instance, nothing else magic of tl.program_id(axis=0)
"""
@triton.jit
def weighted_sum_kernel(
        x_ptr,
        w_ptr,
        output_ptr,
        rows,
        row_num_tiles,
        row_size_tiles : tl.constexpr,
        cols,
        col_num_tiles,
        col_size_tiles : tl.constexpr,

    ):

    pid = tl.program_id(axis=0)

    """
    The following method constructs a block pointer for safe and efficient tensor memory access.

    The `tl.make_block_ptr` function creates a structured pointer that enables
    tiled memory access with automatic boundary checking. This abstraction allows
    kernels to operate on memory blocks while preventing out-of-bounds
    accesses.

    Parameters Explained:
        x_ptr: Base pointer to the tensor data in GPU memory.
    
        shape=(rows, cols): The complete dimensions of the tensor, defining the
            valid memory region that can be safely accessed. This ensures the
            block pointer respects tensor boundaries.
    
        strides=(cols, 1): Memory layout configuration for row-major (C-contiguous)
            tensors. The tuple (cols, 1) indicates:
            - Move `cols` elements to advance one row (row stride)
            - Move 1 element to advance one column (column stride)
    
        offsets=(pid * row_size_tiles, 0): Starting coordinates for this block's
            access window, relative to the tensor origin (0,0). Each program
            instance loads a distinct block beginning at a different row offset
            based on its program ID, enabling parallel processing of row tiles.
    
        block_shape=(row_size_tiles, col_size_tiles): The dimensions of the
            contiguous memory block that will be loaded. This defines the tile
            size each program instance processes in the loop (see following impl code).
    
        order=(1, 0): Memory access pattern within the block. The tuple (1, 0)
            specifies column-major ordering within the block.

        The returned block pointer can be passed to `tl.load()` or `tl.store()`
        to perform memory operations. The pointer automatically adjusts
        indices to remain within the tensor's valid memory region if set boundary_check
        and padding_option.
    """
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(rows, cols),
        strides=(cols, 1),
        offsets=(pid * row_size_tiles, 0),
        block_shape=(row_size_tiles, col_size_tiles),
        order=(1, 0),
    )

    """
    Similar methods.
    """
    w_block_ptr = tl.make_block_ptr(
        w_ptr,
        shape=(cols,),
        strides=(1,),
        offsets=(0,),
        block_shape=(col_size_tiles,),
        order=(0,),
    )
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(rows,),
        strides=(1,),
        offsets=(pid * row_size_tiles,),
        block_shape=(row_size_tiles,),
        order=(0,),
    )

    """
    Column-wise tiled computation loop.

    Iterates over column tiles to process the entire row length while maintaining
    efficient memory access patterns. Each iteration:
    1. Loads a tile of input data from both X and W matrices
    2. Performs fused multiply-reduce operation
    3. Accumulates partial results in on-chip memory
    4. Advances block pointers to next column tile

    Boundary handling: Automatic padding with zeros for partial tiles at edges
    """
    output = tl.zeros((row_size_tiles,), dtype=tl.float32)
    for _ in range(col_num_tiles):
        # Load input tile with boundary checking and zero padding
        # boundary_check=(0,1): Check both row and column boundaries for X
        # boundary_check=(0,): Check only column boundaries for W (1D tensor)
        # padding_option="zero": Pad out-of-bound accesses with zeros
        x_i = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        w_i = tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero")
        
        """
        Fused operation with broadcasting.
    
        x_tile * w_tile[None,:]: Broadcasts w_tile across rows of x_tile
        tl.sum(..., axis=1): Reduces along columns, producing row-wise sums
        output += ...: Accumulates partial results TO memory(may be L2? But definately not main GPU memory)
    
        This fused operation avoids storing intermediate products to SLOWER global memory
        """
        output += tl.sum(x_i * w_i[None,:],axis = 1)

        # Advance block pointers to next column tile
        # x_block_ptr moves (0 rows, col_size_tiles columns)
        # w_block_ptr moves (col_size_tiles elements) for 1D tensor
        x_block_ptr = x_block_ptr.advance((0,col_size_tiles))
        w_block_ptr = w_block_ptr.advance((col_size_tiles,))
    
    """
    Write final results to global memory.
    
    boundary_check=(0,): Ensures row boundary safety for the output store
    """
    tl.store(output_block_ptr, output, boundary_check=(0,))


def kweighted_sum(x, weight):
    cols = x.shape[-1]
    orginal_shape = x.shape[:-1]

    x = x.view(-1, cols).contiguous()
    rows = x.shape[0]

    output = torch.empty(size=(rows,)).to(device=DEVICE)

    assert output.is_contiguous() and weight.is_contiguous() and x.device == DEVICE and weight.device == DEVICE

    COL_SIZE_TILES = 1024 // 16
    COL_NUM_TILES = triton.cdiv(cols, COL_SIZE_TILES)
    
    ROW_SIZE_TILES = 16
    ROW_NUM_TILES = triton.cdiv(rows, ROW_SIZE_TILES)

    weighted_sum_kernel[(ROW_NUM_TILES,)](x,weight,output,rows,ROW_NUM_TILES,ROW_SIZE_TILES, cols, COL_NUM_TILES, COL_SIZE_TILES)

    return output.view(*orginal_shape)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(8, 24, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.randn(size=(size,1024), device=DEVICE, dtype=torch.float32)
    w = torch.randn(size=(1024,), device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: weighted_sum(x, w), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: kweighted_sum(x, w), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True)
"""
vector-add-performance:
         size       Triton        Torch
0       256.0    64.104335   236.307695
1       512.0   132.753546   439.838928
2      1024.0   268.773759   774.047204
3      2048.0   528.871543  1274.606193
4      4096.0  1012.790760  1801.676952
5      8192.0  1779.257961  2068.197162
6     16384.0  2864.309634  2136.317774
7     32768.0  4209.739794  2345.370317
8     65536.0  5749.559928  2504.062166
9    131072.0  7003.151235  2596.154629
10   262144.0  7898.877477  2646.457406
11   524288.0  8403.664852  2674.583483
12  1048576.0  8340.994952  2679.397757
13  2097152.0  8326.290805  2683.638911
14  4194304.0  8317.390256  2683.487000
15  8388608.0  5730.565823  2684.458591
"""







