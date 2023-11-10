"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""
from __future__ import annotations

from pathlib import Path

import torch
from pytorch_transformers.modeling_bert import BertConfig

from src.hand_pose.backbone.hrnet.cfg import cfg as hrnet_config
from src.hand_pose.backbone.hrnet.hrnet_gridfeat import get_cls_net_gridfeat
from src.hand_pose.model.graphormer import Graphormer
from src.hand_pose.utils.datatypes import BackboneArch
from src.utils.config import (
    HAND_TEMPLATE,
    HRNET_W64_CONFIG_PATH,
    HRNET_W64_PATH,
    NUM_JOINTS,
)


class CamParamFC(torch.nn.Module):
    """
    FC predicting camera parameters, using the 3D joint prediction as input
    """

    def __init__(self, num_joints: int, hidden_layer_width: int = 150) -> None:
        super().__init__()
        self.layer_1 = torch.nn.Linear(3, 1)
        self.layer_2 = torch.nn.Linear(num_joints, hidden_layer_width)
        self.layer_3 = torch.nn.Linear(hidden_layer_width, 3)

    def forward(self, pred_3d_joints: torch.Tensor) -> torch.Tensor:
        x = self.layer_1(pred_3d_joints)
        x = x.transpose(1, 2)
        x = self.layer_2(x)
        x = self.layer_3(x)
        cam_param = x.transpose(1, 2)
        cam_param = cam_param.squeeze()

        return cam_param


class GraphormerHandNetwork(torch.nn.Module):
    """
    End-to-end Graphormer network for hand pose and mesh reconstruction from a single image.

    Adapted from https://github.com/microsoft/MeshGraphormer/blob/main/src/modeling/bert/e2e_hand_network.py
    All credit to Microsoft and Lin et al
    """

    def __init__(
        self,
        backbone_arch: BackboneArch = BackboneArch.HRNET_W64,
        bert_config_file: Path = Path("src/hand_pose/model/bert_config.json"),
        cam_fc_width: int = 150,
        input_feat_dim_encoder: tuple[int, ...] = (2051, 512, 128),
        hidden_feat_dim_encoder: tuple[int, ...] = (1024, 256, 64),
        gcn_layers_encoder: tuple[bool, ...] = (False, False, True),
    ) -> None:
        super().__init__()

        # not learnable constants
        self.hand_template = torch.tensor(HAND_TEMPLATE).to("cuda")  # TODO: do nice
        self.num_joints = NUM_JOINTS

        # learnable
        self.backbone = self.backbone_network(backbone_arch)
        self.trans_encoder = self.encoder(
            bert_config_file,
            list(input_feat_dim_encoder),
            list(hidden_feat_dim_encoder),
            list(gcn_layers_encoder),
        )
        self.cam_params_fc = CamParamFC(self.num_joints, cam_fc_width)

        self.grid_feat_dim = torch.nn.Linear(1024, 2051)

    @staticmethod
    def backbone_network(arch: BackboneArch) -> torch.nn.Module:
        if arch == BackboneArch.HRNET_W64:
            hrnet_yaml = HRNET_W64_CONFIG_PATH
            hrnet_checkpoint = HRNET_W64_PATH
            hrnet_config.defrost()
            hrnet_config.merge_from_file(hrnet_yaml)
            hrnet_config.freeze()
            backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)

        else:
            raise NotImplementedError(f"{arch.value} backbone is not implemented")

        return backbone

    @staticmethod
    def encoder(
        config_file: Path,
        input_feat_dim: list[int],
        hidden_feat_dim: list[int],
        gcn_layers: list[bool],
    ) -> torch.nn.Module:
        assert len(input_feat_dim) == len(hidden_feat_dim)

        output_feat_dim = input_feat_dim[1:] + [3]

        trans_encoder: list[torch.nn.Module] = []

        for layer_num in range(len(input_feat_dim)):
            config = BertConfig.from_pretrained(config_file)

            config.output_attentions = False
            config.img_feature_dim = input_feat_dim[layer_num]
            config.output_feature_dim = output_feat_dim[layer_num]
            config.hidden_size = hidden_feat_dim[layer_num]
            config.intermediate_size = 2 * hidden_feat_dim[layer_num]
            config.graph_conv = gcn_layers[layer_num]  # add graph conv

            assert config.hidden_size % config.num_attention_heads == 0

            encoding_layer = Graphormer(config)

            trans_encoder.append(encoding_layer)

        trans_encoder = torch.nn.Sequential(*trans_encoder)

        return trans_encoder

    def forward(
        self,
        images: torch.Tensor,
        is_train: bool = False,
    ):
        # get batch size
        batch_size = images.size(0)

        # concatinate template joints and template vertices, and then duplicate to batch size
        ref_pose = self.hand_template.expand(batch_size, -1, -1)

        # extract grid features and global image features using a CNN backbone
        image_feat, grid_feat = self.backbone(images)
        # concatinate image feat and mesh template
        image_feat = image_feat.view(batch_size, 1, 2048).expand(
            -1,
            ref_pose.shape[-2],
            -1,
        )
        # process grid features
        grid_feat = torch.flatten(grid_feat, start_dim=2)
        grid_feat = grid_feat.transpose(1, 2)
        grid_feat = self.grid_feat_dim(grid_feat)
        # concatinate image feat and template mesh to form the joint/vertex queries
        features = torch.cat([ref_pose, image_feat], dim=2)
        # prepare input tokens including joint/vertex queries and grid features
        features = torch.cat([features, grid_feat], dim=1)

        # TODO: decide if use or not
        # if is_train == True:
        #     # apply mask vertex/joint modeling
        #     # meta_masks is a tensor of all the masks, randomly generated in dataloader
        #     # we pre-define a [MASK] token, which is a floating-value vector with 0.01s
        #     special_token = torch.ones_like(features[:, :-49, :]).cuda() * 0.01
        #     features[:, :-49, :] = features[:, :-49, :] * meta_masks + special_token * (
        #         1 - meta_masks
        #     )

        # forward pass
        features = self.trans_encoder(features)

        pred_3d_joints = features[:, :-49, :]

        # learn camera parameters
        cam_param = self.cam_params_fc(pred_3d_joints)

        return cam_param, pred_3d_joints
