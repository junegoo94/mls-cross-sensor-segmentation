"""config 로드 + 모델/데이터 빌드 + GPU 1-step forward/backward 스모크 테스트."""
import sys
cfg_path = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/jmgoo/projects/os_seg/configs/semseg-pt-v3-coord.py"
sys.argv = ["smoke"]  # 이후 argparse 충돌 방지 (인자 읽은 뒤 초기화)
import torch
from pointcept.utils.config import Config
from pointcept.models import build_model
from pointcept.datasets import build_dataset
from pointcept.datasets.utils import point_collate_fn
print("config:", cfg_path)
cfg = Config.fromfile(cfg_path)

# 데이터셋 빌드 + 샘플 로드
ds = build_dataset(cfg.data.train)
print("train scenes:", len(ds))
s0 = ds[0]
print("sample keys:", sorted(s0.keys()))
for k in ["coord", "grid_coord", "feat", "segment"]:
    if k in s0:
        print(f"  {k}: {tuple(s0[k].shape)} {s0[k].dtype}")
print("feat dim =", s0["feat"].shape[1], "| expected in_channels =",
      cfg.model.backbone.in_channels)
assert s0["feat"].shape[1] == cfg.model.backbone.in_channels, "feat dim != in_channels"

# 모델 빌드 + GPU 1-step
model = build_model(cfg.model).cuda().train()
n_param = sum(p.numel() for p in model.parameters())
print(f"model params: {n_param/1e6:.1f}M")

batch = point_collate_fn([ds[0], ds[1]])
batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
with torch.cuda.amp.autocast(enabled=True):
    out = model(batch)
loss = out["loss"]
print("forward OK | loss =", float(loss))
loss.backward()
print("backward OK")
print("\n>>> SMOKE TEST PASSED")
