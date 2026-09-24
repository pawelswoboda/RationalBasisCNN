"""Graph matching config system."""

import os
from easydict import EasyDict as edict 

__C = edict()
# Consumers can get config by:

cfg = __C

__C.combine_classes = False

__C.BACKBONE_DIR = "./utils/checkpoints"
# VOC2011-Keypoint Dataset
# TODO: Hard-coded to absolute paths for debugging. Normal when running the code
__C.VOC2011 = edict()
__C.VOC2011.KPT_ANNO_DIR = "./data/downloaded/PascalVOC/annotations/"  # keypoint annotation
__C.VOC2011.ROOT_DIR = "./data/downloaded/PascalVOC/VOC2011/"  # original VOC2011 dataset
__C.VOC2011.SET_SPLIT = "./data/split/voc2011_pairs.npz"  # set split path
__C.VOC2011.CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# Willow-Object Dataset
__C.WILLOW = edict()
__C.WILLOW.ROOT_DIR = "./data/downloaded/WILLOW/WILLOW-ObjectClass"
__C.WILLOW.CLASSES = ["Car", "Duck", "Face", "Motorbike", "Winebottle"]
__C.WILLOW.KPT_LEN = 10
__C.WILLOW.TRAIN_NUM = 20
__C.WILLOW.TRAIN_OFFSET = 0

# SPair Dataset
__C.SPair = edict()
__C.SPair.ROOT_DIR = "./data/downloaded/SPair-71k"
__C.SPair.size = "large"
__C.SPair.CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "train",
    "tvmonitor",
]


#
# Training options
#

__C.TRAIN = edict()
__C.TRAIN.difficulty_params = {}
# Stop after this many epochs instead of the length of cfg.TRAIN.lr_schedule.
# 0 uses the schedule's own length. The LR milestones are unchanged, so this
# truncates the reported recipe rather than replacing it.
__C.TRAIN.max_epochs = 0
# Iterations per epochs

__C.EVAL = edict()
__C.EVAL.difficulty_params = {}

# The continuous-kernel convolution of model/sconv_archs.py. input_features /
# output_features come from the experiment json; the rest selects the basis.
#
#   conv = "spline"    B-spline hats, i.e. the SplineConv of SplineCNN. Served
#                      by rational_cnn's pure-PyTorch BSplineConv, so no PyG
#                      C++ extension is needed.
#   conv = "rational"  learnable rational (safe-Pade) basis functions in place
#                      of the hats (RationalBasisCNN); the options below are
#                      the flags of that project's experiments/backbones.py.
__C.SPLINE_CNN = edict()
__C.SPLINE_CNN.conv = "spline"
__C.SPLINE_CNN.dim = 2                 # 2-D keypoint pseudo-coordinates
__C.SPLINE_CNN.kernel_size = 5         # K = kernel_size ** dim = 25 hats
__C.SPLINE_CNN.aggr = "max"            # NMT aggregates neighbours with max
# PyG SplineConv's root/bias initialisation, so the spline baseline starts
# exactly like the torch_geometric.nn.SplineConv that NMT originally used.
__C.SPLINE_CNN.pyg_init = True
# Rational basis options, ignored when conv == "spline".
__C.SPLINE_CNN.rational_basis = "product"   # product | multivariate | mlp
__C.SPLINE_CNN.degrees = [5, 4]             # numerator / denominator degree
__C.SPLINE_CNN.safe = "B"
__C.SPLINE_CNN.poly = "chebyshev"
__C.SPLINE_CNN.init = "spline"              # spline | pca | random | ...
__C.SPLINE_CNN.init_noise = 1e-3
__C.SPLINE_CNN.vp = False                   # variance-preserving weight init
__C.SPLINE_CNN.num_bases = 0                # K for multivariate/mlp; 0 = k**d

# torch.compile the two normalized-transformer decoders (dynamic shapes, the
# padded keypoint count varies per batch). They consist of thousands of tiny
# kernels, so fusing them roughly halves the step time on a single GPU. Same
# fp32 maths; results change only by floating-point rounding, i.e. by less
# than the run-to-run nondeterminism of the atomics in the backward pass.
# Off by default: the logged runs in ../results/logs/nmt_* did not use it.
__C.compile = False

# Keep only the newest epoch in <model_dir>/params instead of every epoch.
# A full sweep otherwise writes ~1.3 GB per epoch per run (model + Adam state).
__C.keep_last_checkpoint_only = False

# Visual backbone: "swin" is the SwinV2-L of the paper, "vgg16" / "vgg16_bn"
# are the much smaller VGG16 of SplineCNN/DGMC (relu4_2 + relu5_1), which makes
# the results comparable to DGMC-scale graph matching models.
__C.BACKBONE = "swin"

# Side length images are resized to. Must match the backbone: 384 for SwinV2
# (its pretrained window sizes), 256 for VGG16 (the DGMC/ThinkMatch setting).
# It also sets the scale of the edge pseudo-coordinates in utils/build_graphs.py
# and the coordinate frame of utils/feature_align.py, which all have to agree.
__C.IMAGE_SIZE = 384

# Mean and std to normalize images
__C.NORM_MEANS = [0.485, 0.456, 0.406]
__C.NORM_STD = [0.229, 0.224, 0.225]

# Data cache path
__C.CACHE_PATH = "data/cache"


__C.train_sampling = "intersection"
__C.eval_sampling = "intersection"
# random seed used for data loading
__C.RANDOM_SEED = 123
