import torch
import torch.distributed as dist

class DDPWarper(torch.nn.Module):
    # See: https://pytorch.org/blog/straggler-mitigation/ 
    # See: https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
    def __init__(self, module: torch.nn.Module): # accepts an instance
        super().__init__()
        self.model = module
        self.handles = []
        self.world_size = dist.get_world_size()
        if dist.is_available() and dist.is_initialized() and (self.world_size is not None):
            for param in self.model.parameters():
                dist.broadcast(param.data, src = 0)
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(lambda p: self._queue_handlers(p))
        else:
            raise RuntimeError("you idiot why not init process group")
        # dist.barrier(process_group)

    def forward(self, *inputs, **kwargs): 
        return self.model(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        # remove latter
        torch.cuda.synchronize()
        
        for param in self.model.parameters():
            if param.requires_grad:
                param.grad = param.grad / self.world_size
        
    def _queue_handlers(self, p):
        #! must be handle = dist.all_reduce(p.grad, async_op=True)
        #! instead of handle = dist.all_reduce(p, async_op=True)
        handle = dist.all_reduce(p.grad, async_op=True)
        self.handles.append(handle)
    
    @property
    def module(self):
        return self.model
