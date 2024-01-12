from setuptools import setup, find_packages

setup(
    name='rdcanon',
    version='0.1',
    packages=find_packages(),
    description='SMARTS Sanitization',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Babak Mahjour',
    author_email='bmahjour@mit.edu',
    url='https://github.com/coleygroup/rdcanon',
    install_requires=[
        'rdkit > 2023.09.1',
        'matplotlib',
        'lark',
        'numpy',
        'networkx',
        'scikit-learn',
        'ipykernel',
        'pandas',
        'openpyxl'
    ],
)