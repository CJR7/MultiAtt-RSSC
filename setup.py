from setuptools import setup, find_packages

setup(
    name='multi_modal_scene_classification',
    version='0.1',
    description='A multi-modal scene classification model integrating Vision Transformers and CLIP models',
    author='Jinjin Cai',
    author_email='cai379@purdue.edu',
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.0',
        'torchvision>=0.8.1',
        'transformers>=4.5.0',
        'Pillow>=8.1.0',
        'numpy>=1.19.2',
        'matplotlib>=3.4.3',
        'scikit-learn',
        'seaborn',
    ],
    python_requires='>=3.6',
)
