# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    difficulty_range=(0.0,1.0),
    sub_terrains={
        #V1
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),

        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.02, 0.1), noise_step=0.02, border_width=0.25    #(0.01, 0.05)
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.10), platform_width=2.0
        ),
        "pyramid_stairs_inv_2": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.25),
            step_width=0.2,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
         "pyramid_stairs_inv_25": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.25),
            step_width=0.25,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
         "pyramid_stairs_inv_3": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
         "pyramid_stairs_inv_35": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.25),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_2": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.25),
            step_width=0.2,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
         "pyramid_stairs_25": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.25),
            step_width=0.25,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
         "pyramid_stairs_3": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
         "pyramid_stairs_35": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.25),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),


        #Play
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.3, grid_width=0.45, grid_height_range=(0.05, 0.25), platform_width=2.0
        # ),
        # #"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.05),
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.3,
        #     step_height_range=(0.15, 0.20),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.4,
        #     step_height_range=(0.05, 0.20),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
    },
)
"""Rough terrains configuration."""