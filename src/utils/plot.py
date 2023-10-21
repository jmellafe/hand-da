from __future__ import annotations

import plotly.graph_objects as go
import torch


def plot_3d_pose_interactive(pose: torch.Tensor) -> None:
    pose = pose.cpu().numpy()[0]

    fig = go.Figure(
        data=[go.Scatter3d(x=pose[:, 0], y=pose[:, 1], z=pose[:, 2], mode="markers")],
    )
    fig.show()
