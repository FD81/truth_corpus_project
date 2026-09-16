import os
import time

# Use faster, more reliable transfer backend
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
user = os.environ["USER"]
os.environ['HF_HOME'] = f'storage/{user}/hf_home/'

from huggingface_hub import snapshot_download

MODEL_ID = "openai/gpt-oss-20b"

def download_with_retry(max_retries=10):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt}/{max_retries}...")
            path = snapshot_download(
                repo_id=MODEL_ID,
                resume_download=True,      # picks up where it left off
                max_workers=4,             # parallel file downloads
            )
            print(f"✅ Download complete: {path}")
            return path
        except Exception as e:
            print(f"❌ Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                wait = min(30, attempt * 5)
                print(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError("Max retries exceeded.")

if __name__ == "__main__":
    download_with_retry()