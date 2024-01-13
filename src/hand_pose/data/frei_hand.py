#!/usr/bin/env python
# mypy: ignore-errors

from __future__ import annotations

import json
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw, ImageSequence
from torch.utils.data import Dataset

from src.utils.config import ADJ_INDICES, FINGER_COLORS, JOINT_NAME, KNUCKLES_INDICES


class DatasetSplit(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"


HandJoint = Enum(
    "HandJoint", {joint_name: idx for idx, joint_name in enumerate(JOINT_NAME)}
)


@dataclass
class CameraParameters:
    intrinsic: np.ndarray


@dataclass
class HandPoseABC(ABC):
    joints: np.ndarray

    def __getitem__(self, key: HandJoint) -> np.ndarray:
        return self.joints[key.value]


@dataclass
class HandPose2D(HandPoseABC):
    def overlay_on_img(self, img: Image.Image, *, in_place: bool = False) -> np.ndarray:
        if not in_place:
            img = img.copy()

        # plot lines connecting joints
        draw = ImageDraw.Draw(img)
        joints_as_list = list(HandJoint)

        for join_idx_1, join_idx_2 in filter(lambda x: x[0] < x[1], ADJ_INDICES):
            color = FINGER_COLORS[joints_as_list[join_idx_2].name.split("_")[0]]

            draw.line(
                tuple(self[joints_as_list[join_idx_1]].astype(int))
                + tuple(self[joints_as_list[join_idx_2]].astype(int)),
                fill=tuple(color),
            )

        return img


@dataclass
class HandPose3D(HandPoseABC):
    joints: np.ndarray

    def project_2d(self, cam_params: CameraParameters) -> HandPose2D:
        projected_unnormalized = cam_params.intrinsic.dot(self.joints.T).T

        projected_normalized = (
            projected_unnormalized[:, :2] / projected_unnormalized[:, 2:]
        )

        return HandPose2D(projected_normalized)

    def plot(
        self,
        out_path: Path,
        elev: int = 10,
        gif_frame_duration: int = 20,
    ) -> None:
        # define fig and 3d axis
        fig = plt.figure()
        ax = Axes3D(fig)

        # plot 3d hand
        ## plot lines connecting joints
        joints_as_list = list(HandJoint)

        for join_idx_1, join_idx_2 in filter(lambda x: x[0] < x[1], ADJ_INDICES):
            color = FINGER_COLORS[joints_as_list[join_idx_2].name.split("_")[0]]

            ax.plot(
                [self.joints[join_idx_1, 0], self.joints[join_idx_2, 0]],
                [self.joints[join_idx_1, 1], self.joints[join_idx_2, 1]],
                [self.joints[join_idx_1, 2], self.joints[join_idx_2, 2]],
                marker="o",
                c=np.array(color) / 255,
            )

        ## plot line between knuckles to make the plot more clear
        ax.plot(
            self.joints[KNUCKLES_INDICES, 0],
            self.joints[KNUCKLES_INDICES, 1],
            self.joints[KNUCKLES_INDICES, 2],
            c="gray",
        )
        ax.set_axis_off()

        # animate 3d plot
        def animate(i: int) -> plt.Figure:
            ax.view_init(elev=elev, azim=i)

        anim = animation.FuncAnimation(
            fig, animate, frames=360, interval=gif_frame_duration
        )

        anim.save(out_path)

        plt.cla()


@dataclass
class ImageAnnotation:
    hand_pose_3d: HandPose3D
    hand_pose_2d: HandPose2D | None = None


def plot_annotation(
    annotation: ImageAnnotation,
    img: Image.Image,
    plot_path: Path,
    gif_frame_duration: int = 20,
) -> None:
    annotation.hand_pose_3d.plot(plot_path, gif_frame_duration=gif_frame_duration)

    if annotation.hand_pose_2d is not None:
        img_with_overlay = annotation.hand_pose_2d.overlay_on_img(img)

        # add to generated gif
        gif_cap = Image.open(plot_path)

        concatted_frames: list[Image.Image] = []
        for frame in ImageSequence.Iterator(gif_cap):
            # resize gif frame to match hand image height
            frame = frame.resize(
                (
                    int(frame.width * img_with_overlay.height / frame.height),
                    img_with_overlay.height,
                )
            )

            # concat images side by side
            concat_img = Image.new(
                "RGB", (img_with_overlay.width + frame.width, img_with_overlay.height)
            )
            concat_img.paste(img_with_overlay, (0, 0))
            concat_img.paste(frame, (img_with_overlay.width, 0))

            # concat_img.save("sup.png")

            concatted_frames.append(concat_img.convert("P"))

        concatted_frames[0].save(
            plot_path,
            format="GIF",
            save_all=True,
            append_images=concatted_frames[1:],
            duration=gif_frame_duration,
            loop=0,
            optimize=False,
            lossless=True,
        )


class FreiHANDDataset(Dataset):
    def __init__(self, dataset_path: Path, split: DatasetSplit) -> None:
        self.dataset_path = dataset_path
        self.split = split

        self.camera_parameters = self._load_camera_parameters(dataset_path, split)
        self.annotations = self._load_annotations(
            dataset_path, split, self.camera_parameters
        )

        img_folder_name = "evaluation" if split == DatasetSplit.TESTING else "training"
        self.img_dir = dataset_path / img_folder_name / "rgb"

    @staticmethod
    def _load_camera_parameters(
        dataset_path: Path, split: DatasetSplit
    ) -> list[CameraParameters]:
        assert split in (
            DatasetSplit.TRAINING,
            DatasetSplit.TESTING,
        ), "Camera parameters only available for training and testing set (val needs to be added)"

        file_name = (
            "evaluation_K.json" if split == DatasetSplit.TESTING else "training_K.json"
        )

        with open(dataset_path / file_name) as f:
            camera_parameters: list[list[list[float]]] = json.load(f)

        camera_parameters = [
            CameraParameters(np.array(cam_params)) for cam_params in camera_parameters
        ]

        return camera_parameters

    @staticmethod
    def _load_annotations(
        dataset_path: Path, split: DatasetSplit, cam_parameters: list
    ) -> list[ImageAnnotation]:
        assert (
            split == DatasetSplit.TRAINING
        ), "Annotations only available for training set (val needs to be added)"

        with open(dataset_path / "training_xyz.json") as f:
            joints_3d: list[list[list[float]]] = json.load(f)

        joints_3d = [HandPose3D(np.array(joints)) for joints in joints_3d]

        annotations = [
            ImageAnnotation(j_3d, j_3d.project_2d(cam_params))
            for j_3d, cam_params in zip(joints_3d, cam_parameters)
        ]

        return annotations

    def load_image(self, index: int) -> Image.Image:
        img = Image.open(self.img_dir / f"{index:08d}.jpg")

        return img

    def __getitem__(self, index: int) -> tuple:
        self.load_image(index)
        self.annotations[index]

        # plot annotation for debugging
        # plot_annotation(img_annotation, img, Path("check.gif"))

        print("hye")


if __name__ == "__main__":
    ds = FreiHANDDataset(Path("/data/FreiHAND_pub_v2"), DatasetSplit.TRAINING)

    out = ds[0]

    print("hye")
