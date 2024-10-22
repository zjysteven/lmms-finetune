import re
from typing import Dict, List, Sequence, Union
import PIL

import PIL.Image
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
from . import register_collator
from .base import BaseDataCollator
import json

SYSTEM_MESSAGE = "You are a helpful assistant."
DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
IGNORE_INDEX = -100


@register_collator("qwen2-vl")
class Qwen2VLDataCollator(BaseDataCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if "images" in instances[0]:
            is_video = False
        elif "videos" in instances[0]:
            is_video = True

        # # print(instances[0])
        #{'images': [<PIL.Image.Image image mode=RGB size=1368x912 at 0x7CAC21C72FB0>, <PIL.Image.Image image mode=RGB size=650x431 at 0x7CAC21C73190>], 
        # 'videos': [], 
        # 'conversations': ['<image><image>Which image has more people? Give details to your answer.', 'There are five people in the first image, while there are eleven people in the second. Therefore, the second image has more people.'], '
        # system_prompt': 'Answer the following questions by considering all images.'}
        if not is_video:
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            videos = None
            # images: List[PIL.Image.Image] = [x for instance in instances for x in instance["images"]]
            images: List[List[PIL.Image.Image]] = [instance["images"] for instance in instances]
            # print(images)
        else:
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"
            images = None
            videos: List[np.ndarray] = [x for instance in instances for x in instance["videos"]]

        # texts
        # the dataset implementation assume conversations are [user, assistant, user, assistant, ...]
        system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        conversations: List[List] = [instance["conversations"] for instance in instances]
        max_len = self.tokenizer.model_max_length

        total_image_tokens = 0
        batch_input_ids = []
        batch_labels = []
        batch_pixel_values = []
        batch_vision_grid_thw = []
        
        for b_idx, (system_prompt, cur_convs) in enumerate(zip(system_prompts, conversations)):
            cur_input_ids = []
            cur_labels = []
            cur_pixel_values = []
            cur_vision_grid_thw = []            
            cur_text = []

            if system_prompt is None:
                system_prompt = SYSTEM_MESSAGE
            
            for i, text in enumerate(cur_convs):
                if i % 2 == 0:
                    num_image_tokens = len([m.start() for m in re.finditer("<image>", text)])
                    total_image_tokens += num_image_tokens

                    cur_text.append({
                        "role": "user",
                        "content": replace_image_tokens(text, is_video=is_video)
                    })
                else:
                    cur_text.append({
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": text},
                        ]
                    })
            
            # heavily borrowed from https://github.com/2U1/Qwen2-VL-Finetune
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}\n{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = self.processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
            cur_input_ids.append(system_message_input_ids.squeeze(0))
            cur_labels.append(system_labels.squeeze(0))
            for idx, j in enumerate(range(0, len(cur_text), 2)):
                # user输入和assistant输出处理
                user_input = cur_text[j]
                gpt_response = cur_text[j + 1]
                user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n"
                gpt_response = f"{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n{gpt_response['content']}\n{DEFAULT_IM_END_TOKEN}\n"
                
                if idx == 0:
                    if not is_video:
                        # print("Input images", images[b_idx]) # 貌似只能读一张图片
                        # 这里也不能直接改成images送进去，不然所有对话处理的image都是从images的第一张开始算到text对应image的数量的
                        # 这里需要处理的是一个cur_text对应的所有图片
                        inputs = self.processor(text=[user_input], images=images[b_idx], videos=None, padding=False, return_tensors='pt')
                    else:
                        inputs = self.processor(text=[user_input], images=None, videos=videos[b_idx], padding=False, return_tensors='pt')
                    prompt_input_ids = inputs['input_ids']
                    # 对于780 * 1040图片, 图像token数量大致为1034， prompt_input为1057
                    # 所以image token是算在input ids中
                    pixel_values = inputs[pixel_key] # 单图: torch.Size([4144, 1176]) torch.Size([1 * height//14 * width//14 , 1176])
                    # print(pixel_values.shape)
                    vision_grid_thw = inputs[grid_key] # 单图: tensor([[ 1, 74, 56]]), torch.Size([1, 3])
                    # print(vision_grid_thw.shape)
                    # print(prompt_input_ids.shape)
                else:
                    prompt_input_ids = self.processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

                response_input_ids = self.processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

                input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
                # 拼接后去掉第一个维度(必须是1),不是1不会进行squeeze操作
                if self.mask_question_tokens:
                    # 不计算question的loss，只计算response的loss,一般是用这种做法
                    labels = torch.cat(
                        [
                            torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                            response_input_ids.squeeze(0),
                        ],
                        dim=0,
                    )
                else:
                    labels = cur_input_ids.clone()
                cur_input_ids.append(input_ids)
                cur_labels.append(labels)
                cur_pixel_values.append(pixel_values)
                cur_vision_grid_thw.append(vision_grid_thw)
            
            cur_input_ids = torch.cat(cur_input_ids, dim=0).to(torch.long)
            cur_labels = torch.cat(cur_labels, dim=0).to(torch.long)
            cur_pixel_values = torch.cat(cur_pixel_values, dim=0)
            cur_vision_grid_thw = torch.cat(cur_vision_grid_thw, dim=0)
            # manual truncation
            # 超过的手动截断
            # 这里vision tokens是参与token数量计算的，所以max_len不能太小，否则会导致回答的token被手动截断
            if cur_input_ids.shape[0] > max_len:
                cur_input_ids = cur_input_ids[:max_len]
                cur_labels = cur_labels[:max_len]

            assert cur_input_ids.shape == cur_labels.shape, "Input and label shapes do not match"
            
            cur_input_ids = cur_input_ids.unsqueeze(0)
            cur_labels = cur_labels.unsqueeze(0)
            
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

            batch_input_ids.append(cur_input_ids)
            batch_labels.append(cur_labels)
            batch_pixel_values.append(cur_pixel_values)
            batch_vision_grid_thw.append(cur_vision_grid_thw)
            
        batch_input_ids = torch.cat(batch_input_ids, dim=0)
        batch_labels = torch.cat(batch_labels, dim=0)
        batch_pixel_values = torch.cat(batch_pixel_values, dim=0)
        batch_vision_grid_thw = torch.cat(batch_vision_grid_thw, dim=0)

        # sanity check
        assert total_image_tokens == count_innermost_elements(images), "Number of image tokens does not match the number of images"

        data_dict = dict(
            input_ids=batch_input_ids,
            labels=batch_labels,
            attention_mask=batch_input_ids.ne(self.PAD_TOKEN_ID),       
        )
        data_dict[pixel_key] = batch_pixel_values
        data_dict[grid_key] = batch_vision_grid_thw
        
        return data_dict

def count_innermost_elements(nested_list):
    # 如果当前元素不是列表，说明是最内层元素，返回 1
    if not isinstance(nested_list, list):
        return 1
    # 如果是列表，递归统计所有子列表的元素数量
    return sum(count_innermost_elements(item) for item in nested_list)

def _findall(token_list: torch.Tensor, token: int) -> torch.Tensor:
    if not isinstance(token_list, torch.Tensor):
        raise ValueError("token_list must be a PyTorch Tensor")
    mask = token_list == token
    indices = torch.where(mask)[0]

    return indices


def replace_image_tokens(input_string, is_video=False):
    if is_video:
        input_string = input_string.replace("<video>"+'\n', "<|vision_start|>"+"<|video_pad|>"+"<|vision_end|>")
        input_string = input_string.replace("<video>", "<|vision_start|>"+"<|video_pad|>"+"<|vision_end|>")

    else:
        input_string = input_string.replace("<image>"+'\n', "<|vision_start|>"+"<|image_pad|>"+"<|vision_end|>")
        input_string = input_string.replace("<image>", "<|vision_start|>"+"<|image_pad|>"+"<|vision_end|>")

    return input_string

if __name__ == "__main__":
    model_path = '../models/qwen2-vl-7b-instruct'
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    datacollator = Qwen2VLDataCollator(config=config, tokenizer=tokenizer, processor=processor)
    with open('../datasets/Q-Instruct_Q-Align/test.json', 'r') as f:
        data = json.load(f)
    
    
    # print(datacollator(data))