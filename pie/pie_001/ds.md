# Cooking Pies
## DeepSpeed Overview

DeepSpeed analyzes your network topology using a `hostfile`. A typical `hostfile` contains the following configurations:

```txt
worker-0 slots=8
worker-1 slots=8
worker-2 slots=8
```

In this configuration:
- `worker-i` represents server aliases
- IP addresses and ports should be configured in the `~/.ssh/config` file
- DeepSpeed uses passwordless SSH to connect to each machine
- One process is created for each available GPU
- If no `hostfile` is provided, DeepSpeed defaults to single-node mode
- Use `-H hostfile` to specify the hostfile explicitly

### SSH Configuration Example

```txt
Host worker-0
    HostName localhost
    Port 22
```

### Connection Verification

To verify SSH connectivity, use:

```bash
ssh -vvv -o PasswordAuthentication=no worker-0 hostname
```

After analyzing the network topology and establishing connections, DeepSpeed launches your Python code. Example launch command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed deepspeed_test.py --deepspeed
```

Refer to log files in the GitHub directory for detailed launch information.

## Model Modifications for DeepSpeed

### Local Rank Handling

DeepSpeed automatically passes `--local-rank` to your Python script. Proper handling is required:

- All Python processes connect to the master host
- The master node manages and coordinates all other nodes
- Use `deepspeed.init_distributed()` on all processes for automatic setup
- Environment variables `LOCAL_RANK` and `RANK` are automatically set by DeepSpeed
- This API call is blocking (Not tested)

### Accelerator Abstraction

DeepSpeed provides accelerator abstractions for hardware-agnostic model execution. For example:

```python
_local_rank = int(os.environ.get('LOCAL_RANK'))
get_accelerator().set_device(_local_rank)
```

### Migration Cheatsheet

| Original Code | DeepSpeed Equivalent |
|---------------|---------------------|
| `tensor.cuda()` | `tensor.to(get_accelerator().device_name())` |
| `tensor.is_cuda` | `get_accelerator().on_accelerator(tensor)` |
| `tensor.pin_memory()` | `get_accelerator().pin_memory(tensor)` |
| `torch.cuda.current_device()` | `get_accelerator().current_device_name()` |
| `torch.cuda.default_generators[index]` | `get_accelerator().default_generator(index)` |
| `torch.distributed.init_process_group('nccl')` | `torch.distributed.init_process_group(get_accelerator().communication_backend_name())` |

For detailed information, refer to the [DeepSpeed Accelerator Abstraction Interface documentation](https://www.deepspeed.ai/tutorials/accelerator-abstraction-interface/).

### Model Initialization

During initializing optimizer, scheduler, dataloader, and model, we can

- Use `torch.distributed.barrier()` for synchronization
- Processes calling this API block until all processes invoke it

Then, wrap components using DeepSpeed initialization:

```python
engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=model.parameters(),
    training_data=dataset,
    config=ds_config,
)
```

### Training Process

Writing training code using DeepSpeed is easy:

- Move data to `local_device = get_accelerator().device_name()`
- No explicit scheduler management required but use `engine.backward(loss)` and `engine.step()` APIs
- For model checkpointing: `engine.save_checkpoint(save_dir=args.save_dir)`. **Important**: All processes must call this API to avoid blocking

### Argument Parsing



## GPU Utilization Measurement

### FLOPs Calculation

To calculate model FLOPs (especially for input-dependent models):

```python
from torch.utils.flop_counter import FlopCounterMode

flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
with flop_counter:
    model(**input).sum().backward()
total_flops = flop_counter.get_total_flops()
tflops = total_flops / time / 1e12
```

### Hardware FLOPs Utilization (HFU)

- Run multiple iterations for input-dependent FLOPs calculations
- Calculate FLOPS using total FLOPs and training steps
- Note: Techniques like gradient checkpointing may affect actual HFU
- This method provides a good approximation of Hardware FLOPS Utilization