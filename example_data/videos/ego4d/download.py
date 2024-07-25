from huggingface_hub import hf_hub_download

hf_hub_download(
    "ShareGPT4Video/ShareGPT4Video",
    "zip_folder/ego4d/ego4d_videos_4.zip",
    repo_type="dataset",
    local_dir="."
)