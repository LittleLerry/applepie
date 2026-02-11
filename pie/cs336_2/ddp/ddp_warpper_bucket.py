import torch
import torch.distributed as dist

class DDPWarper(torch.nn.Module):

    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()

        self.model = module
        self.bucket_capacity_bytes = int(bucket_size_mb * 1024**2)
        # TODO check bucket_size_bytes has proper size

        self.cur_bucket_size_bytes = 0
        self.cur_bucket = [] # store tensors
        self.handles = []

        self.world_size = dist.get_world_size()
        if dist.is_available() and dist.is_initialized() and (self.world_size is not None):
            for param in self.model.parameters():
                dist.broadcast(param.data, src = 0)
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(lambda p: self._queue_handlers(p))
        else:
            raise RuntimeError("you idiot why not init process group")

    def forward(self, *inputs, **kwargs): 
        return self.model(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):

        for (handle, flattened, l) in self.handles:
            handle.wait()
            # data ready
            grads = [param.grad for param in l]
            unflattened = torch._utils._unflatten_dense_tensors(flattened, grads)
            for (p, reduced) in zip(l, unflattened):
                p.grad = reduced / self.world_size
        
        if len(self.cur_bucket) > 0:
            grads = [param.grad for param in self.cur_bucket]
            flattened = torch._utils._flatten_dense_tensors(grads)
            dist.all_reduce(flattened, async_op=False)
            unflattened = torch._utils._unflatten_dense_tensors(flattened, grads)
            for (p, reduced) in zip(self.cur_bucket, unflattened):
                p.grad = reduced / self.world_size
        
        self._clean()
        torch.cuda.synchronize()
        
    def _queue_handlers(self, p):
        grad_size_in_bytes = p.grad.numel() * p.grad.element_size()
        if ((grad_size_in_bytes + self.cur_bucket_size_bytes > self.bucket_capacity_bytes) and (len(self.cur_bucket) > 0)):

            grads = [param.grad for param in self.cur_bucket]
            flattened = torch._utils._flatten_dense_tensors(grads)
            handle = dist.all_reduce(flattened, async_op=True)
            # unflattened = torch._utils._unflatten_dense_tensors(flattened, grads)
            self.handles.append((handle, flattened, self.cur_bucket.copy()))
            # clear
            self.cur_bucket_size_bytes = 0
            self.cur_bucket.clear()

        self.cur_bucket.append(p)
        self.bucket_capacity_bytes = self.cur_bucket_size_bytes + grad_size_in_bytes        
        """
        handle = dist.all_reduce(p.grad, async_op=True)
        self.handles.append(handle)
        """
    def _clean(self):
        self.cur_bucket.clear()
        self.cur_bucket_size_bytes = 0
        self.handles.clear()

    @property
    def module(self):
        return self.model
