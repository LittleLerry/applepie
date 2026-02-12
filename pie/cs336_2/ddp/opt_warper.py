import torch
from typing import Type, Any
import torch.distributed as dist

class optWarper(torch.optim.Optimizer):
    """
    I don't like the following shit code. 
    Basically we init a super class using super().__init__(params, kwargs),
    which automatically analyzes the parameters' groups. Based on this, we pass corespondding 
    list to sub-optimizers. Though self holds the total ref of params, it does not update them.
    """
    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
        # params: ref to tesnors
        super().__init__(params, kwargs)

        self.world_size = dist.get_world_size()
        if dist.is_available() and dist.is_initialized() and (self.world_size is not None):
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

            local_paras = []
            assert len(list(self._parameters_g())) >= self.world_size

            for (idx , p) in enumerate(list(self._parameters_g())):
                if idx % self.world_size ==  self.rank:
                    local_paras.append(p)

            self.opt = optimizer_cls(local_paras, **kwargs) # init with partial parameters

        else:
            raise RuntimeError("you idiot why not init process group")
        dist.barrier()

    def _parameters_g(self):
        for group in self.param_groups:
            for p in group["params"]:
                if (p.requires_grad):
                    yield p
    
    def step(self):
        # update all grads
        dist.barrier()
        for p in self._parameters_g():
            dist.all_reduce(p.grad)
            p.grad = p.grad / self.world_size
        torch.cuda.synchronize()
        # restore weights
        self.opt.step()
        for (idx , p) in enumerate(list(self._parameters_g())):
            dist.broadcast(p.data, src=idx % self.world_size)
