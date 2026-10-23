"""Upload the weights-only checkpoints to Hugging Face (run `hf auth login` first)."""
import glob, os, sys
from huggingface_hub import HfApi
REPO = sys.argv[1] if len(sys.argv) > 1 else "jmgoo1118/mls-cross-sensor-segmentation"
D = "/home/jmgoo/projects/os_seg/exp/release_weights"
api = HfApi()
api.create_repo(REPO, repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj=f"{D}/HF_README.md", path_in_repo="README.md", repo_id=REPO)
for f in sorted(glob.glob(f"{D}/*.pth")):            # top level only: private/ is never uploaded
    print("uploading", os.path.basename(f), flush=True)
    api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f), repo_id=REPO)
print("done: https://huggingface.co/" + REPO)
