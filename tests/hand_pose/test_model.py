from __future__ import annotations

import pytest
import torch

from src.hand_pose.model.model import GraphormerHandNetwork


@pytest.mark.parametrize(
    "input_shape, output_shape_cam, output_shape_joints",
    [
        ((1, 3, 224, 224), (3,), (1, 21, 3)),
        ((2, 3, 224, 224), (2, 3), (2, 21, 3)),
        ((5, 3, 224, 224), (5, 3), (5, 21, 3)),
    ],
)
@torch.no_grad()
def test_hand_pose_model(
    hand_pose_model: GraphormerHandNetwork,
    device: str,
    input_shape: tuple[int, int, int, int],
    output_shape_cam: tuple[int, ...],
    output_shape_joints: tuple[int, ...],
) -> None:
    input_img = torch.zeros(input_shape).to(device)

    cam_param, pred_3d_joints = hand_pose_model(input_img)

    # breakpoint()
    assert cam_param.shape == output_shape_cam, "Wrong predicted cam param shape"
    assert pred_3d_joints.shape == output_shape_joints, "Wrong predicted 3d joint shape"
