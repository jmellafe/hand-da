# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import annotations

from yacs.config import CfgNode as CN


cfg = CN()

cfg.OUTPUT_DIR = ""
cfg.LOG_DIR = ""
cfg.DATA_DIR = ""
cfg.GPUS = (0,)
cfg.WORKERS = 4
cfg.PRINT_FREQ = 20
cfg.AUTO_RESUME = False
cfg.PIN_MEMORY = True
cfg.RANK = 0

# Cudnn related params
cfg.CUDNN = CN()
cfg.CUDNN.BENCHMARK = True
cfg.CUDNN.DETERMINISTIC = False
cfg.CUDNN.ENABLED = True

# common params for NETWORK
cfg.MODEL = CN()
cfg.MODEL.NAME = "cls_hrnet"
cfg.MODEL.INIT_WEIGHTS = True
cfg.MODEL.PRETRAINED = ""
cfg.MODEL.NUM_JOINTS = 17
cfg.MODEL.NUM_CLASSES = 1000
cfg.MODEL.TAG_PER_JOINT = True
cfg.MODEL.TARGET_TYPE = "gaussian"
cfg.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
cfg.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
cfg.MODEL.SIGMA = 2
cfg.MODEL.EXTRA = CN(new_allowed=True)

cfg.LOSS = CN()
cfg.LOSS.USE_OHKM = False
cfg.LOSS.TOPK = 8
cfg.LOSS.USE_TARGET_WEIGHT = True
cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

# DATASET related params
cfg.DATASET = CN()
cfg.DATASET.ROOT = ""
cfg.DATASET.DATASET = "mpii"
cfg.DATASET.TRAIN_SET = "train"
cfg.DATASET.TEST_SET = "valid"
cfg.DATASET.DATA_FORMAT = "jpg"
cfg.DATASET.HYBRID_JOINTS_TYPE = ""
cfg.DATASET.SELECT_DATA = False

# training data augmentation
cfg.DATASET.FLIP = True
cfg.DATASET.SCALE_FACTOR = 0.25
cfg.DATASET.ROT_FACTOR = 30
cfg.DATASET.PROB_HALF_BODY = 0.0
cfg.DATASET.NUM_JOINTS_HALF_BODY = 8
cfg.DATASET.COLOR_RGB = False

# train
cfg.TRAIN = CN()

cfg.TRAIN.LR_FACTOR = 0.1
cfg.TRAIN.LR_STEP = [90, 110]
cfg.TRAIN.LR = 0.001

cfg.TRAIN.OPTIMIZER = "adam"
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WD = 0.0001
cfg.TRAIN.NESTEROV = False
cfg.TRAIN.GAMMA1 = 0.99
cfg.TRAIN.GAMMA2 = 0.0

cfg.TRAIN.BEGIN_EPOCH = 0
cfg.TRAIN.END_EPOCH = 140

cfg.TRAIN.RESUME = False
cfg.TRAIN.CHECKPOINT = ""

cfg.TRAIN.BATCH_SIZE_PER_GPU = 32
cfg.TRAIN.SHUFFLE = True

# testing
cfg.TEST = CN()

# size of images for each device
cfg.TEST.BATCH_SIZE_PER_GPU = 32
# Test Model Epoch
cfg.TEST.FLIP_TEST = False
cfg.TEST.POST_PROCESS = False
cfg.TEST.SHIFT_HEATMAP = False

cfg.TEST.USE_GT_BBOX = False

# nms
cfg.TEST.IMAGE_THRE = 0.1
cfg.TEST.NMS_THRE = 0.6
cfg.TEST.SOFT_NMS = False
cfg.TEST.OKS_THRE = 0.5
cfg.TEST.IN_VIS_THRE = 0.0
cfg.TEST.COCO_BBOX_FILE = ""
cfg.TEST.BBOX_THRE = 1.0
cfg.TEST.MODEL_FILE = ""

# debug
cfg.DEBUG = CN()
cfg.DEBUG.DEBUG = False
cfg.DEBUG.SAVE_BATCH_IMAGES_GT = False
cfg.DEBUG.SAVE_BATCH_IMAGES_PRED = False
cfg.DEBUG.SAVE_HEATMAPS_GT = False
cfg.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, config_file):
    cfg.defrost()
    cfg.merge_from_file(config_file)
    cfg.freeze()


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "w") as f:
        print(cfg, file=f)
