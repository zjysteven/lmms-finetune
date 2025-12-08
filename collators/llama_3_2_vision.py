import re
from typing import Dict, List, Sequence, Union

import numpy as np
import PIL
import torch
from transformers.image_utils import get_image_size, to_numpy_array
from transformers.models.mllama.processing_mllama import MllamaProcessorKwargs
from transformers.utils import logging

from . import register_collator
from .base import BaseDataCollator
from .chat_template_monkey_patch import apply_chat_template


logger = logging.get_logger(__name__)


# slightly different from https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/blob/main/chat_template.json
# to add "{% generation %}" keyword to return a mask of the assistant generated tokens
template = (
    "{{- bos_token }}\n"
    "{%- if custom_tools is defined %}\n"
    "{%- set tools = custom_tools %}\n"
    "{%- endif %}\n"
    "{%- if not tools_in_user_message is defined %}\n"
    "{%- set tools_in_user_message = true %}\n"
    "{%- endif %}\n"
    "{%- if not date_string is defined %}\n"
    "{%- if strftime_now is defined %}\n"
    "{%- set date_string = strftime_now(\"%d %b %Y\") %}\n"
    "{%- else %}\n"
    "{%- set date_string = \"26 Jul 2024\" %}\n"
    "{%- endif %}\n"
    "{%- endif %}\n"
    "{%- if not tools is defined %}\n"
    "{%- set tools = none %}\n"
    "{%- endif %}\n\n"
    "{#- This block extracts the system message, so we can slot it into the right place. #}\n"
    "{%- if messages[0]['role'] == 'system' %}\n"
    "{%- set system_message = messages[0]['content']|trim %}\n"
    "{%- set messages = messages[1:] %}\n"
    "{%- set user_supplied_system_message = true %}\n"
    "{%- else %}\n"
    "{%- set system_message = \"\" %}\n"
    "{%- set user_supplied_system_message = false %}\n"
    "{%- endif %}\n\n"
    "{#- Find out if there are any images #}\n"
    "{% set image_ns = namespace(has_images=false) %}\n"
    "{%- for message in messages %}\n"
    "{%- for content in message['content'] %}\n"
    "{%- if content['type'] == 'image' %}\n"
    "{%- set image_ns.has_images = true %}\n"
    "{%- endif %}\n"
    "{%- endfor %}\n"
    "{%- endfor %}\n\n"
    "{#- System message if there are no images, or if the user supplied one #}\n"
    "{%- if user_supplied_system_message or not image_ns.has_images %}\n"
    "{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n"
    "{%- if tools is not none %}\n"
    "{{- \"Environment: ipython\\n\" }}\n"
    "{%- endif %}\n"
    "{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n"
    "{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n"
    "{%- if tools is not none and not tools_in_user_message %}\n"
    "{{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n"
    "{{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n"
    "{{- \"Do not use variables.\\n\\n\" }}\n"
    "{%- for t in tools %}\n"
    "{{- t | tojson(indent=4) }}\n"
    "{{- \"\\n\\n\" }}\n"
    "{%- endfor %}\n"
    "{%- endif %}\n"
    "{{- system_message }}\n"
    "{{- \"<|eot_id|>\" }}\n"
    "{%- endif %}\n\n"
    "{#- Custom tools are passed in a user message with some extra guidance #}\n"
    "{%- if tools_in_user_message and not tools is none %}\n"
    "{#- Extract the first user message so we can plug it in here #}\n"
    "{%- if messages | length != 0 %}\n"
    "{%- set first_user_message = messages[0]['content']|trim %}\n"
    "{%- set messages = messages[1:] %}\n"
    "{%- else %}\n"
    "{{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n"
    "{%- endif %}\n"
    "{{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n"
    "{{- \"Given the following functions, please respond with a JSON for a function call \" }}\n"
    "{{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n"
    "{{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n"
    "{{- \"Do not use variables.\\n\\n\" }}\n"
    "{%- for t in tools %}\n"
    "{{- t | tojson(indent=4) }}\n"
    "{{- \"\\n\\n\" }}\n"
    "{%- endfor %}\n"
    "{{- first_user_message + \"<|eot_id|>\"}}\n"
    "{%- endif %}\n\n"
    "{%- for message in messages %}\n"
    "{%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}\n"
    "{%- if message['content'] is string %}\n"
    "{{- message['content'] }}\n"
    "{%- else %}\n"
    "{%- for content in message['content'] %}\n"
    "{%- if content['type'] == 'image' %}\n"
    "{{- '<|image|>' }}\n"
    "{%- elif content['type'] == 'text' and message.role != 'assistant' %}\n"
    "{{- content['text'] }}\n"
    "{%- elif content['type'] == 'text' and message.role == 'assistant' %}\n"
    "{% generation %}"
    "{{- content['text'] }}\n"
    "{% endgeneration %}"
    "{%- endif %}\n"
    "{%- endfor %}\n"
    "{%- endif %}\n"
    "{{- '<|eot_id|>' }}\n"
    "{%- elif 'tool_calls' in message %}\n"
    "{%- if not message.tool_calls|length == 1 %}\n"
    "{{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n"
    "{%- endif %}\n"
    "{%- set tool_call = message.tool_calls[0].function %}\n"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n"
    "{{- '{\"name\": \"' + tool_call.name + '\", ' }}\n"
    "{{- '\"parameters\": ' }}\n"
    "{{- tool_call.arguments | tojson }}\n"
    "{{- \"}\" }}\n"
    "{{- \"<|eot_id|>\" }}\n"
    "{%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n"
    "{{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n"
    "{%- if message.content is mapping or message.content is iterable %}\n"
    "{{- message.content | tojson }}\n"
    "{%- else %}\n"
    "{{- message.content }}\n"
    "{%- endif %}\n"
    "{{- \"<|eot_id|>\" }}\n"
    "{%- endif %}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n"
    "{%- endif %}\n"
)


@register_collator("llama-3.2-vision")
class LLaMA3_2_VisionDataCollator(BaseDataCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # monkey patch to include bos tokens
        self.tokenizer.apply_chat_template = apply_chat_template.__get__(self.tokenizer)

        output_kwargs = self.processor._merge_kwargs(
            MllamaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
        )
        
        vision_inputs = dict()
        images: List[List[PIL.Image.Image]] = [x for instance in instances for x in instance["images"]]
        if len(images) > 0:
            image_features = self.processor.image_processor(images, return_tensors="pt", **output_kwargs["images_kwargs"])
            num_tiles = image_features.pop("num_tiles")
            vision_inputs.update(**image_features)

        # constants
        max_len = self.tokenizer.model_max_length
        image_token_id = self.config.image_token_index
        
        input_ids = []
        labels = []
        
        # some parsing
        images: List[List[PIL.Image.Image]] = [instance["images"] for instance in instances]
        system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        conversations: List[List] = [instance["conversations"] for instance in instances]
        
        for system_prompt, cur_images, cur_convs in zip(system_prompts, images, conversations):
            cur_num_images = 0
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

                    # .strip(): whitespaces and newlines are handled by chat_template
                    text = text.replace("<image>", "").strip()

                    cur_text.append({
                        "role": "user",
                        "content": [{"type": "text", "text": text}] + \
                            [{"type": "image"}] * num_images
                    })
                else:
                    cur_text.append({
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": text},
                        ]
                    })

            assert len(cur_images) == cur_num_images, "Number of image tokens does not match the number of images"

            temp = self.tokenizer.apply_chat_template(
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

            # a dirty hack to include eos token as part of the labels
            cur_assistant_masks[0, -1] = True
            
            # manual truncation
            if cur_input_ids.shape[1] > max_len:
                cur_input_ids = cur_input_ids[:, :max_len]
                cur_assistant_masks = cur_assistant_masks[:, :max_len]
            cur_labels = cur_input_ids.clone()

            # mask question tokens
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