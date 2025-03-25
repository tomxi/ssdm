from setuptools import setup, find_packages

setup(
    name='ssdm',
    version='0.1.0',
    description='Scanning SDMs for MSA representation selection',
    author='Tom Xi',
    author_email='tom.xi@nyu.edu',
    url='https://github.com/tomxi/ssdm',
    packages=find_packages(),
    # install_requires=[
    #     'openl3',
    #     'jams',
    #     'mir_eval',
    # ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Collaborators',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
)