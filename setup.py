from setuptools import setup, find_packages

setup(
    name='VideoProcessing',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'os-sys',
        'imageio',
        'h5py',
        'numpy',
        'Pillow',
        'logging',
        'torch',
        'torchvision',
        'wandb',
        'opencv-python',
        'av'
    ],
    entry_points={
        'console_scripts': [
            'video_process=convert:video_to_h5',
        ],
    },
    python_requires='>=3.6',
)


