from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv

load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
if not HUGGING_FACE_TOKEN:
    raise ValueError("HUGGING_FACE_TOKEN not found in .env file")

print("Start download")
snapshot_download(
    repo_id="LGAI-DILab/ReTabAD",
    repo_type="dataset",
    local_dir="./data",
    token=HUGGING_FACE_TOKEN
)
print("Done")