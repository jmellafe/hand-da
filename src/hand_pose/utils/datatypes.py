from __future__ import annotations

from enum import Enum


class BackboneArch(Enum):
    # backbone for hand pose model
    # TODO: implement more backbones
    HRNET_W64 = "hrnet-w64"
