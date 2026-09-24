from setuptools import setup, find_packages

setup(
    name='rational_cnn',
    version='0.1.0',
    description=('Learnable rational (safe-Padé) basis functions for '
                 'continuous-kernel graph convolutions (SplineCNN)'),
    author='Paul Swoboda',
    author_email='paul.swoboda@hhu.de',
    python_requires='>=3.9',
    install_requires=['torch', 'torch_geometric>=2.4', 'numpy', 'scipy'],
    extras_require={'faust': ['plyfile'], 'voc': ['torchvision'],
                    'test': ['pytest']},
    # nmt/ is a separate code base that runs from its own directory; it only
    # imports rational_cnn and must not be installed as part of it.
    packages=find_packages(exclude=['tests', 'experiments', 'nmt', 'nmt.*']),
)
