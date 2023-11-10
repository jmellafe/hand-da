from __future__ import annotations

from pathlib import Path

import pytest

from src.hand_pose.model.model import GraphormerHandNetwork


@pytest.fixture
def device() -> str:
    return "cuda"


@pytest.fixture
def hand_pose_model_path() -> Path | None:
    # TODO: replace by trained model path once we have one
    return None


@pytest.fixture
def hand_pose_model(
    hand_pose_model_path: Path | None,
    device: str,
) -> GraphormerHandNetwork:
    model = GraphormerHandNetwork()

    # TODO: load from checkpoint

    model = model.to(device)

    return model
