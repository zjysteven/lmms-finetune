import re
from typing import Dict, List, Sequence, Union

import numpy as np
import PIL
import torch
from transformers.image_utils import get_image_size, to_numpy_array
from transformers.utils import logging

from . import register_collator
from .base import BaseDataCollator
from .chat_template_monkey_patch import apply_chat_template


logger = logging.get_logger(__name__)


@register_collator("llava-next-video")
class LLaVANeXTVideoDataCollator(BaseDataCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # monkey patch to include bos tokens
        self.tokenizer.apply_chat_template = apply_chat_template.__get__(self.tokenizer)

        vision_inputs = dict()

        # images
        images: List[PIL.Image.Image] = [x for instance in instances for x in instance["images"]]
        if len(images) > 0:
            vision_inputs.update(**self.processor.image_processor(images, return_tensors="pt"))

        # videos
        videos: List[np.ndarray] = [x for instance in instances for x in instance["videos"]]
        if len(videos) > 0:
            # ideally we should do padding here instead of forcing all videos to have the same length
            # but since currently hf implementation does not unpad videos or have corresponding 
            # attention masks, having padding will let the model train on padded frames
            assert len(set([x.shape[0] for x in videos])) == 1, "All videos must have the same number of frames"
            vision_inputs.update(**self.processor.video_processor(videos, return_tensors="pt"))

        # some parsing
        images = [instance["images"] for instance in instances]
        videos = [instance["videos"] for instance in instances]
        system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        conversations: List[List] = [instance["conversations"] for instance in instances]

        # constants
        max_len = self.tokenizer.model_max_length
        image_token_id = self.config.image_token_index
        video_token_id = self.config.video_token_index
        patch_size = self.processor.patch_size
        vision_feature_select_strategy = self.processor.vision_feature_select_strategy

        input_ids = []
        labels = []
        
        for system_prompt, cur_images, cur_videos, cur_convs in zip(system_prompts, images, videos, conversations):
            cur_num_images = 0
            cur_num_videos = 0
            cur_input_ids = []
            cur_labels = []
            
            cur_text = []
            if system_prompt is not None:
                cur_text.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                })
            
            for i, text in enumerate(cur_convs):
                if i % 2 == 0:
                    num_images = len([m.start() for m in re.finditer("<image>", text)])
                    cur_num_images += num_images

                    num_videos = len([m.start() for m in re.finditer("<video>", text)])
                    cur_num_videos += num_videos

                    # .strip(): whitespaces and newlines are handled by chat_template
                    text = text.replace("<image>", "").replace("<video>", "").strip()

                    cur_text.append({
                        "role": "user",
                        "content": [{"type": "text", "text": text}] + \
                            [{"type": "image"}] * num_images + \
                            [{"type": "video"}] * num_videos
                    })
                else:
                    cur_text.append({
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": text},
                        ]
                    })

            assert len(cur_images) == cur_num_images, "Not all images were used"
            assert len(cur_videos) == cur_num_videos, "Not all videos were used"

            temp = self.processor.apply_chat_template(
                cur_text,
                add_generation_prompt=False,
                tokenize=True,
                return_assistant_tokens_mask=True,
                return_dict=True,
                return_tensors="pt",
                truncation=False, # the assistant tokens mask seems wrong when truncation is enabled
            )
            cur_input_ids = temp["input_ids"]
            cur_assistant_masks = torch.tensor(temp["assistant_masks"], dtype=torch.bool).unsqueeze(0)

            # expand vision tokens
            if patch_size is None or vision_feature_select_strategy is None:
                logger.warning_once(
                    "Expanding inputs for image/video tokens in LLaVa-NeXT-Video should be done in processing. "
                    "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
                    "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
                    "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
                )
            else:
                if len(cur_images) > 0:
                    image_inputs = self.processor.image_processor(cur_images, return_tensors="pt")
                    image_sizes = image_inputs["image_sizes"]
                    height, width = get_image_size(to_numpy_array(image_inputs["pixel_values"][0][0]))

                    num_image_tokens_list = []
                    for image_size in image_sizes:
                        if not isinstance(image_size, (list, tuple)):
                            # cast to list to avoid numerical precision errors when calculating unpadding
                            orig_height, orig_width = image_size.tolist()
                        num_image_tokens = self.processor._get_number_of_features(orig_height, orig_width, height, width)
                        if vision_feature_select_strategy == "default":
                            num_image_tokens -= 1
                        num_image_tokens_list.append(num_image_tokens)

                    repeat = torch.ones(cur_input_ids.shape[1], dtype=torch.long)
                    repeat[torch.where(cur_input_ids == image_token_id)[1]] = torch.tensor(num_image_tokens_list, dtype=torch.long)
                    cur_input_ids = cur_input_ids.repeat_interleave(repeat, dim=1)
                    cur_assistant_masks = cur_assistant_masks.repeat_interleave(repeat, dim=1)
                
                if len(cur_videos) > 0:
                    video_inputs = self.processor.video_processor(cur_videos, return_tensors="pt")

                    one_video = to_numpy_array(video_inputs.get("pixel_values_videos")[0])
                    height, width = get_image_size(one_video[0])
                    num_frames = one_video.shape[0]  # frame dim is always after batch dim
                    num_image_tokens = (height // patch_size) * (width // patch_size)
                    num_video_tokens = num_image_tokens // 4 * num_frames  # divide by 4 needed for avg pooling layer

                    repeat = torch.where(cur_input_ids == video_token_id, num_video_tokens, 1).squeeze()
                    cur_input_ids = cur_input_ids.repeat_interleave(repeat, dim=1)
                    cur_assistant_masks = cur_assistant_masks.repeat_interleave(repeat, dim=1)

            # a dirty hack to include eos token as part of the labels
            cur_assistant_masks[0, -1] = True

            # manual truncation
            if cur_input_ids.shape[1] > max_len:
                cur_input_ids = cur_input_ids[:, :max_len]
                cur_assistant_masks = cur_assistant_masks[:, :max_len]
            cur_labels = cur_input_ids.clone()

            if self.mask_question_tokens:
                assert cur_labels.shape == cur_assistant_masks.shape, "Label and mask shapes do not match"
                cur_labels = torch.where(cur_assistant_masks, cur_labels, self.IGNORE_TOKEN_ID)
            
            assert cur_input_ids.shape == cur_labels.shape, "Input and label shapes do not match"

            # padding
            if cur_input_ids.shape[1] < max_len:
                cur_input_ids = torch.cat([
                    cur_input_ids,
                    torch.full(
                        (cur_input_ids.shape[0], max_len - cur_input_ids.shape[1]),
                        self.PAD_TOKEN_ID,
                        dtype=cur_input_ids.dtype,
                        device=cur_input_ids.device
                    )
                ], dim=1)
                cur_labels = torch.cat([
                    cur_labels,
                    torch.full(
                        (cur_labels.shape[0], max_len - cur_labels.shape[1]),
                        self.IGNORE_TOKEN_ID,
                        dtype=cur_labels.dtype,
                        device=cur_labels.device
                    )
                ], dim=1)

            input_ids.append(cur_input_ids)
            labels.append(cur_labels)

        input_ids = torch.cat(input_ids)
        labels = torch.cat(labels)

        return dict(
            **vision_inputs,
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.PAD_TOKEN_ID),
        )