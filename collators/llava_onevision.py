import math
import re
from typing import Dict, List, Sequence, Union

import numpy as np
import PIL
import torch
from transformers.image_utils import get_image_size, to_numpy_array
from transformers.models.llava_onevision.processing_llava_onevision import LlavaOnevisionProcessorKwargs
from transformers.utils import logging

from . import register_collator
from .base import BaseDataCollator


logger = logging.get_logger(__name__)


# slightly different from https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf/blob/main/chat_template.json
# to include <|im_end|> of assistant's response as labels
template = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + ' '}}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>' }}"
    "{% endfor %}"
    "{# Render all video then #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'video') %}"
    "{{ '<video>' }}"
    "{% endfor %}"
    "{# Render all text next #}"
    "{% if message['role'] != 'assistant' %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ '\n' + content['text'] }}"
    "{% endfor %}"
    "{% else %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{% generation %}"
    "{{ '\n' + content['text'] }}"
    "{{'<|im_end|>'}}"
    "{% endgeneration %}"
    "{% endfor %}"
    "{% endif %}"
    "{% if message['role'] != 'assistant' %}"
    "{{'<|im_end|>'}}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


@register_collator("llava-onevision")
class LLaVAOnevisionDataCollator(BaseDataCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        output_kwargs = self.processor._merge_kwargs(
            LlavaOnevisionProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
        )

        vision_inputs = dict()        
        # images
        images: List[List[PIL.Image.Image]] = [x for instance in instances for x in instance["images"]]
        if len(images) > 0:
            vision_inputs.update(**self.processor.image_processor(images, return_tensors="pt", **output_kwargs["images_kwargs"]))

        # videos
        videos: List[np.ndarray] = [x for instance in instances for x in instance["videos"]]
        if len(videos) > 0:
            # ideally we should do padding here instead of forcing all videos to have the same length
            # but since currently hf implementation does not unpad videos or have corresponding 
            # attention masks, having padding will let the model train on padded frames
            assert len(set([x.shape[0] for x in videos])) == 1, "All videos must have the same number of frames"
            vision_inputs.update(**self.processor.video_processor(videos, return_tensors="pt", **output_kwargs["videos_kwargs"]))

        # some parsing
        images = [instance["images"] for instance in instances]
        videos = [instance["videos"] for instance in instances]
        system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        conversations: List[List] = [instance["conversations"] for instance in instances]
        
        # constants
        max_len = self.tokenizer.model_max_length
        image_token_id = self.config.image_token_index
        video_token_id = self.config.video_token_index
        vision_feature_select_strategy = self.processor.vision_feature_select_strategy

        # construct input_ids and labels
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
                        "content": [{"type": "text", "text": text}]
                    })
                
            assert len(cur_images) == cur_num_images, "Not all images were used"
            assert len(cur_videos) == cur_num_videos, "Not all videos were used"
            
            temp = self.processor.apply_chat_template(
                cur_text,
                chat_template=template,
                add_generation_prompt=False,
                tokenize=True,
                return_assistant_tokens_mask=True,
                return_dict=True,
                return_tensors="pt",
                truncation=False # the assistant tokens mask seems wrong when truncation is enabled
            )
            cur_input_ids = temp["input_ids"]
            cur_assistant_masks = torch.tensor(temp["assistant_masks"], dtype=torch.bool).unsqueeze(0)

            # expand vision tokens
            if len(cur_images) > 0:
                image_inputs = self.processor.image_processor(cur_images, return_tensors="pt", **output_kwargs["images_kwargs"])

                image_sizes = image_inputs["image_sizes"]
                height, width = get_image_size(
                    to_numpy_array(image_inputs["pixel_values"][0][0]), 
                    channel_dim=output_kwargs["images_kwargs"].get("data_format")
                )

                num_image_tokens_list = []
                for image_size in image_sizes:
                    orig_height, orig_width = image_size
                    num_image_tokens = self.processor._get_number_of_features(orig_height, orig_width, height, width)
                    if vision_feature_select_strategy == "default":
                        num_image_tokens -= 1
                    num_image_tokens_list.append(num_image_tokens)

                repeat = torch.ones(cur_input_ids.shape[1], dtype=torch.long)
                repeat[torch.where(cur_input_ids == image_token_id)[1]] = torch.tensor(num_image_tokens_list, dtype=torch.long)
                cur_input_ids = cur_input_ids.repeat_interleave(repeat, dim=1)
                cur_assistant_masks = cur_assistant_masks.repeat_interleave(repeat, dim=1)
            
            if len(cur_videos) > 0:
                video_inputs = self.processor.video_processor(cur_videos, return_tensors="pt", **output_kwargs["videos_kwargs"])

                one_video = to_numpy_array(video_inputs["pixel_values_videos"][0])
                height, width = get_image_size(
                    one_video[0], 
                    channel_dim=output_kwargs["images_kwargs"].get("data_format")
                )
                num_frames = one_video.shape[0]  # frame dim is always after batch dim
                patches_height_width = int(math.sqrt(self.processor.num_image_tokens))
                pooled_height_width = math.ceil(patches_height_width / 2)
                num_video_tokens = (num_frames * pooled_height_width * pooled_height_width) + 1  # +1 for newline token

                repeat = torch.where(cur_input_ids == video_token_id, num_video_tokens, 1).squeeze()
                cur_input_ids = cur_input_ids.repeat_interleave(repeat, dim=1)
                cur_assistant_masks = cur_assistant_masks.repeat_interleave(repeat, dim=1)

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