COLLATORS = {}

def register_collator(name):
    def register_collator_cls(cls):
        if name in COLLATORS:
            return COLLATORS[name]
        COLLATORS[name] = cls
        return cls
    return register_collator_cls


from .llava_1_5 import LLaVA15DataCollator
from .llava_1_6 import LLaVA16DataCollator
from .llava_interleave import LLaVAInterleaveDataCollator
from .llava_next_video import LLaVANeXTVideoDataCollator
from .qwen_vl import QwenVLDataCollator
from .phi3_v import Phi3VDataCollator
from .qwen2_vl import Qwen2VLDataCollator