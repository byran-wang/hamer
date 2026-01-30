from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='HaMeR as a package',
    name='hamer',
    packages=find_packages(),
    install_requires=[
        'gdown',
        'numpy==1.26.3',
        'opencv-python',
        'pyrender',
        'pytorch-lightning==2.3.0',
        'scikit-image',
        'smplx==0.1.28',
        'yacs',
        #'detectron2 @ git+https://github.com/facebookresearch/detectron2',
        '--no-build-isolation detectron2 @ git+https://github.com/facebookresearch/detectron2.git@ff53992b1985b63bd3262b5a36167098e3dada02',
        # if there is no torch found when installing detectron2 even though torch is already installed,
        # please install detectron2 following:
        # git clone https://github.com/facebookresearch/detectron2.git
        # git checkout ff53992b1985b63bd3262b5a36167098e3dada02
        # conda install -c conda-forge gcc=11 gxx=11 -n hamer 
        # export CC=$CONDA_PREFIX/bin/gcc                                                   
        # export CXX=$CONDA_PREFIX/bin/g++ 
        # python -m pip install -e detectron2 --no-build-isolation 
        '--no-build-isolation chumpy @ git+https://github.com/mattloper/chumpy',
        'mmcv==1.3.9',
        'timm',
        'einops',
        'xtcocotools',
        'pandas',
    ],
    extras_require={
        'all': [
            'hydra-core',
            'hydra-submitit-launcher',
            'hydra-colorlog',
            'pyrootutils',
            'rich',
            'webdataset',
        ],
    },
)
