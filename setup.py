from setuptools import setup, find_packages


setup(
    name='mmsurv',
    version='0.1.0',
    description='MM-SurvModels',
    url='https://github.com/ezgiogulmus/MM-SurvModels',
    author='FEO',
    author_email='',
    license='GPLv3',
    packages=find_packages(exclude=['assets', 'datasets_csv', "splits"]),
    install_requires=[
        "torch>=2.3.0",
        "numpy==1.23.4", 
        "pandas==1.4.3",
        "h5py",
        "scikit-learn", 
        "scikit-survival",
        "tensorboardx",
        "pot==0.9.3"
    ],

    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: GPLv3",
    ]
)