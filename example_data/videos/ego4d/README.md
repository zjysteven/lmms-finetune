# Download data for reproducing `example.sh`

This will download ~14GB of video clips from ShareGPT4Video.

```bash
python download.py

unzip zip_folder/ego4d/ego4d_videos_4.zip -d .
rm -rf zip_folder
```