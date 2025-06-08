# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains import FlatPatchSamplingCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    # 계단의 갯수는 size 와 step_width, platform_width 로 부터 결정 가능하다.
    size=(10.75, 10.75), # (9.75.9.75)
    border_width=10.0,
    num_rows=20,
    num_cols=10,
    color_scheme="random",
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.5,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.1, 0.18),
            step_width=0.5, #0.5
            platform_width=3.0,
            border_width=3,
            holes=False,
            flat_patch_sampling={
                "root_spawn": FlatPatchSamplingCfg(
                    num_patches=10,
                    patch_radius=0.3,
                    x_range=(-1.0, 1.0),
                    y_range=(-1.0, 1.0),
                    z_range=(-1.0, 1.0),
                    max_height_diff=0.1,
                ),
                
                "target": FlatPatchSamplingCfg(
                    num_patches=10,
                    patch_radius=0.3,
                    x_range=(-7.5, 7.5),
                    y_range=(-7.5, 7.5),
                    z_range=(-1.0, 1.0),
                    max_height_diff=0.1,
                ),
            },
        ),
    },
)
"""Rough terrains configuration."""
