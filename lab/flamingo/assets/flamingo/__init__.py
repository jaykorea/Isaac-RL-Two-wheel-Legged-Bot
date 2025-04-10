# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing asset and sensor configurations."""

import os
import toml

# Conveniences to other module directories via relative paths
FLAMINGO_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

FLAMINGO_ASSETS_DATA_DIR = os.path.join(FLAMINGO_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

FLAMINGO_ASSETS_METADATA = toml.load(os.path.join(FLAMINGO_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = FLAMINGO_ASSETS_METADATA["package"]["version"]


##
# Configuration for different assets.
##

from .flamingo_rev01_5_1 import *
# from .flamingo_rev01_4_3 import * # Depricated
from .flamingo_edu_v1 import *
