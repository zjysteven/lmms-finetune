import torch.distributed as dist

LOADERS = {}

def register_loader(name):
    def register_loader_cls(cls):
        if name in LOADERS:
            return LOADERS[name]
        LOADERS[name] = cls
        return cls
    return register_loader_cls

def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args)


from .llava_1_5 import LLaVA15ModelLoader
from .llava_1_6 import LLaVA16ModelLoader
from .llava_interleave import LLaVAInterleaveModelLoader
from .llava_next_video import LLaVANeXTVideoModelLoader
from .llava_onevision import LLaVAOnevisionModelLoader
from .qwen_vl import QwenVLModelLoader
from .phi3_v import Phi3VModelLoader
from .qwen2_vl import Qwen2VLModelLoader