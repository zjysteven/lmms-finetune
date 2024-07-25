import re
from typing import Dict, List, Sequence, Union
import PIL

import numpy as np
import torch

from . import register_collator
from .base import BaseDataCollator


@register_collator("llava-next-video")
class LLaVANeXTVideoDataCollator(BaseDataCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        vision_inputs = dict()

        # images
        images: List[PIL.Image.Image] = [x for instance in instances for x in instance["images"]]
        if len(images) > 0:
            vision_inputs.update(**self.processor.image_processor(images, return_tensors="pt"))

        # videos
        # we do manual padding here so that each video has the same length
        # so that we can batch them together
        # here has a limitation: huggingface implementation
        # currently consumes all batched frames (no "unpadding" is ever done)
        # so if there are clips that are significantly shorter than the longest clip
        # then the model will be trained on many padded frames
        videos: List[np.ndarray] = [x for instance in instances for x in instance["videos"]]
        if len(videos) > 0:
            max_num_frames = max([x.shape[0] for x in videos])
            for i, video in enumerate(videos):
                if video.shape[0] < max_num_frames:
                    pad = np.zeros((video.shape[1], video.shape[2], video.shape[3]), dtype=np.uint8)
                    pad = np.expand_dims(pad, axis=0).repeat(max_num_frames - video.shape[0], axis=0)
                    videos[i] = np.concatenate([video, pad], axis=0)
            vision_inputs.update(**self.processor.video_processor(videos, return_tensors="pt"))

        # texts
        # the dataset implementation assume conversations are [user, assistant, user, assistant, ...]
        system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        conversations: List[List] = [instance["conversations"] for instance in instances]
        max_len = self.tokenizer.model_max_length

        user_token_id = self.tokenizer(
            "USER:", add_special_tokens=False, padding=False, return_tensors="pt"
        )["input_ids"]
        assistant_token_id = self.tokenizer(
            "ASSISTANT:", add_special_tokens=False, padding=False, return_tensors="pt"
        )["input_ids"]

        total_image_tokens = 0
        total_video_tokens = 0
        input_ids = []
        labels = []
        
        for system_prompt, cur_convs in zip(system_prompts, conversations):
            cur_input_ids = []
            cur_labels = []
            
            cur_text = []
            if system_prompt is not None:
                cur_text.append({
                    "role": "system",
                    # add a space at the end because for now the chat template
                    # adds nothing after the system prompt
                    # and this will affect the slicing of question/answer
                    # if we don't add a space
                    "content": [{"text": system_prompt.rstrip() + " "}]
                })
            
            for i, text in enumerate(cur_convs):
                if i % 2 == 0:
                    num_image_tokens = len([m.start() for m in re.finditer("<image>", text)])
                    total_image_tokens += num_image_tokens

                    num_video_tokens = len([m.start() for m in re.finditer("<video>", text)])
                    total_video_tokens += num_video_tokens

                    # .strip(): whitespaces and newlines are handled by chat_template
                    text = text.replace("<image>", "").replace("<video>", "").strip()

                    cur_text.append({
                        "role": "user",
                        "content": [{"type": "text", "text": text}] + \
                            [{"type": "image"}] * num_image_tokens + \
                            [{"type": "video"}] * num_video_tokens
                    })
                else:
                    cur_text.append({
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": text},
                        ]
                    })

            cur_text = self.processor.apply_chat_template(cur_text, tokenize=False, add_generation_prompt=False)
            cur_input_ids = self.tokenizer(
                cur_text, truncation=True, max_length=max_len, return_tensors="pt"
            )["input_ids"]
            cur_labels = cur_input_ids.clone()
            
            if self.mask_question_tokens:
                # locate each question slice of the conversation
                # by finding the indices of the user and assistant tokens
                user_locs = np.where(np.array(cur_input_ids[0]) == user_token_id[0][0].item())[0]
                assistant_locs = np.where(np.array(cur_input_ids[0]) == assistant_token_id[0][0].item())[0]

                # filter potential false positives
                user_locs_filtered = [
                    i for i in user_locs if cur_input_ids[0, i:i + user_token_id.shape[1]].tolist() == user_token_id[0].tolist()
                ]
                assistant_locs_filtered = [
                    i for i in assistant_locs if cur_input_ids[0, i:i + assistant_token_id.shape[1]].tolist() == assistant_token_id[0].tolist()
                ]
                assert len(user_locs_filtered) == len(assistant_locs_filtered), "Number of user and assistant tokens do not match"

                start_inds = []; end_inds = []
                for i, (user_loc, assistant_loc) in enumerate(zip(user_locs_filtered, assistant_locs_filtered)):
                    start_inds.append(user_loc if i > 0 else 0)
                    end_inds.append(assistant_loc + assistant_token_id.shape[1])

                for start_ind, end_ind in zip(start_inds, end_inds):
                    cur_labels[0, start_ind:end_ind] = self.IGNORE_TOKEN_ID
            
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

        # sanity check
        assert total_image_tokens == len(images), "Number of image tokens does not match the number of images"
        assert total_video_tokens == len(videos), "Number of video tokens does not match the number of videos"

        input_ids = torch.cat(input_ids)
        labels = torch.cat(labels)

        return dict(
            **vision_inputs,
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.PAD_TOKEN_ID),
        )