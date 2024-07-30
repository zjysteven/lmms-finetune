# Dataset json file

```json
[
    {
        "system_prompt": "Answer the following questions about the image and video.",
        "video": ["bm_teaser.mp4", "bm_show.mp4"],
        "image": "bm.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<video><video>What are these videos about?"
            },
            {
                "from": "gpt",
                "value": "These videos are featuring a Kpop group, BabyMonster."
            },
            {
                "from": "human",
                "value": "<image>What does this image show?"
            },
            {
                "from": "gpt",
                "value": "This image shows the members of the Kpop group BabyMonster."
            }
        ]
    }
]
```

The above example shows one dataset entry that has all the keys that the code will look for (but some of them are not required to be presented all the time). Let's go over them one by one.

- `system_prompt`: This is the prompt that will be put at the very beginning of the conversation as a general instruction for the model. If there is no `system_prompt` key presented, then for the current sample there will simply be no system prompt.
- `video`: This is a list of paths (or a single string path) to the video(s). The paths could be relative or absolute, depending on whether the `video_folder` argument is specified to the training script. The only requirement is that the number of `<video>` token in the conversations should be the same as the number of videos in the current sample. If there is no `video` key presented, then the current sample will obviously have no video(s). Currently the script will sample a fixed number of frames from each video, and the number is specified by the `num_frames` argument to the training script. The reason for this (fixed number of frames) is that at the moment huggingface video models (e.g., LLaVA-NeXT-Video) do not unpad the video frames and do not have corresponding attention masks. So if we pad the video frames to the same length, the model will train on padded frames which is not ideal.
- `image`: This is a list of paths (or a single string path) to the image(s). The path could be relative or absolute, depending on whether the `image_folder` argument is specified to the training script. The only requirement is that the number of `<image>` token in the conversations should be the same as the number of images in the current sample. If there is no `image` key presented, then the current sample will obviously have no image(s).
- `conversations`: This is a list of conversation turns, alternating between the human/user and the model/assistant. Please make sure it strictly follows the order of human, model, human, model, human, model, ... Note, the role key is not fixed and can be specified by the `user_key` and `assistant_key` arguments to the training script. For instance, if your dataset uses "user" and "assistant" instead of "human" and "gpt", you can specify `user_key="user"` and `assistant_key="assistant"` to the training script. The `conversations` key is required to be presented in each dataset entry (otherwise there will be nothing to train on).


:warning: **If you have text-only entries in your training dataset**: the training is likely to fail at some point if 1) your `per_device_batch_size` is 1, or 2) the number of text-only instances dominate the number of multi-modal instances. This is due to a limitation/bug of deepspeed. If neither of the above two conditions is met, no worries, we got you covered.
