"""训练目录名 → docs/bbox_data2.json 的物体键。

只在调用方没有传入 ``--bbox-key`` 时使用。不参与拒帧或投票。
"""
from __future__ import annotations

OBJ_DIR_TO_BBOX_KEY: dict[str, str] = {
    "data2_shaver": "electric_shaver",
    "data2_rabbit": "brown_rabbit",
    "data2_golden_retriever": "golden_retriever",
    "data2_bowl": "golden_bowl",
    "data2_umbrella": "umbrella",
    "data2_toy_cake": "toy_cake",
    "data2_cookie": "cookie_bag",
    "data2_medicine_bottle": "medicine_bottle",
    "data2_bracelet": "bracelet",
    "data2_hair_clip": "hair_clip",
    "data2_tape": "double_sided_tape",
}
