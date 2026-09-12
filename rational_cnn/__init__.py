r"""Learnable rational (safe-Padé) basis functions for continuous-kernel graph
convolutions, as a drop-in replacement for SplineCNN's B-spline basis."""
from .bspline import BSplineConv, open_bspline_basis_1d, bspline_basis
from .rational import (RationalBasis1D, RationalBasis,
                       MultivariateRationalBasis, MLPBasis, RationalConv,
                       RationalCNN)
from .spline_cnn import SplineCNN
from .dgmc import DGMC
from .data import PairDataset, ValidPairDataset
from .face_to_edge import FaceToEdge
from .ply import read_ply

__version__ = '0.1.0'

__all__ = [
    'BSplineConv',
    'open_bspline_basis_1d',
    'bspline_basis',
    'RationalBasis1D',
    'RationalBasis',
    'MultivariateRationalBasis',
    'MLPBasis',
    'RationalConv',
    'RationalCNN',
    'SplineCNN',
    'DGMC',
    'PairDataset',
    'ValidPairDataset',
    'FaceToEdge',
    'read_ply',
]
