from setuptools import setup, find_packages

setup(
    name='ssdm',
    version='0.1.0',
    description='Scanning SDMs for MSA representation selection',
    author='Tom Xi',
    author_email='tom.xi@nyu.edu',
    url='https://github.com/tomxi/ssdm',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'seaborn',
        'librosa',
        'openl3',
        'tensorflow >= 2.0',
        'holoviews',
        'panel',
        'scipy',
        'sklearn',
        'torch',
        'tqdm',
        'pandas',
        'jams',
        'mir_eval',
    ],ß
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Collaborators',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)