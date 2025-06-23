# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains import FlatPatchSamplingCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    # 계단의 갯수는 size 와 step_width, platform_width 로 부터 결정 가능하다.
    size=(10.0, 10.0), # (9.75.9.75)
    border_width=7.5,
    num_rows=20,
    num_cols=10,
    color_scheme="random",
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.5,
    difficulty_range=(0.05, 0.9),
    use_cache=True,
    sub_terrains={ #  HfPyramidStairsTerrainCfg
        "hf_pyramid_stair_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            # inverted=True,
            holes=False,
            proportion=0.9,
            step_height_range=(0.02, 0.13),
            step_width=0.4, #0.5
            platform_width=2.5,
            border_width=1.0,
            flat_patch_sampling={
                "root_spawn": FlatPatchSamplingCfg(
                    num_patches=5,
                    patch_radius=0.2,
                    x_range=(-1.0, 1.0),
                    y_range=(-1.0, 1.0),
                    z_range=(-1.0, 1.0),
                    max_height_diff=0.15,
                ),
                "target": FlatPatchSamplingCfg(
                    num_patches=5,
                    patch_radius=0.2,
                    x_range=(-5.0, 5.0),
                    y_range=(-5.0, 5.0),
                    z_range=(0.0, 1.0),
                    max_height_diff=0.15,
                ),
            },
        ),
    },
)
"""Rough terrains configuration."""
