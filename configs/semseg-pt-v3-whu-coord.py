"""
WHU-Urban3D (city A, MLS) PTv3 — coord-only, 통합 11클래스. 단일소스(증강B 없음).
WHU 단일센서(city A) 학습. cross-sensor 평가·3센서 leave-one-out·joint 재사용용.
- 데이터: projects/os_seg/data/joint (DefaultDataset 포맷, 0.05m voxel)
- 9 classes (0..8, incl. Pedestrian) + ignore(-1)
- 야외 MLS 증강 (nuscenes PTv3 기반), CE + Lovasz loss (불균형 대응)
- strength 는 전처리에서 per-tile z-score 정규화됨 (OS_data 전이 대비)
- 짝 config: semseg-pt-v3-coord.py (coord-only, in_channels=3)

* Pointcept tools/train.py 로 실행. RTX 3090 x2 기준.
* enable_flash=True (flash-attn prebuilt wheel 설치됨)
"""
_base_ = ["/home/jmgoo/projects/Pointcept/configs/_base_/default_runtime.py"]

# ---- misc ----
batch_size = 8          # 전체 GPU 합산 (3090 x2 → 4/GPU)
num_worker = 16
mix_prob = 0.8
empty_cache = False
enable_amp = True
enable_wandb = False
evaluate = True

# ---- model (변형에서 in_channels override) ----
model = dict(
    type="DefaultSegmentorV2",
    num_classes=11,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,                       # coord-only
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,                   # flash-attn (prebuilt wheel) 설치됨
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# ---- schedule ----
epoch = 100
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

# ---- dataset ----
dataset_type = "DefaultDataset"
data_root = "/home/jmgoo/projects/os_seg/data/whu3d"
ignore_index = -1
names = ["Ground", "Building", "Natural", "Pole", "Car", "Fence_Barrier",
         "Road_markings", "Utility_line", "Bollard", "Trash_can", "Pedestrian"]

# feat_keys 는 변형 config 에서 정의 (coord-only vs coord+strength)
feat_keys = ("coord",)

data = dict(
    num_classes=11,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="GridSample", grid_size=0.05, hash_type="fnv",
                 mode="train", return_grid_coord=True),
            dict(type="SphereCrop", point_max=102400, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(type="Collect",
                 keys=("coord", "grid_coord", "segment"),
                 feat_keys=feat_keys),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(type="GridSample", grid_size=0.05, hash_type="fnv",
                 mode="train", return_grid_coord=True, return_inverse=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(type="Collect",
                 keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                 feat_keys=feat_keys),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type="GridSample", grid_size=0.05, hash_type="fnv",
                          mode="test", return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(type="Collect",
                     keys=("coord", "grid_coord", "index"),
                     feat_keys=feat_keys),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
            ],
        ),
        ignore_index=ignore_index,
    ),
)
