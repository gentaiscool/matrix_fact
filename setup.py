import os
from setuptools import setup

# read the contents of your README file
from os import path
dir_path = path.abspath(path.dirname(__file__))
with open(path.join(dir_path, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
print(long_description)

setup(name='matrix-fact',
      version='1.1.2',
      description='Python Matrix Factorization Module',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Genta Indra Winata',
      author_email='gentaindrawinata@gmail.com',
      url='https://github.com/gentaiscool/matrix_fact',
      packages = ['matrix_fact'],    
      license = 'GNU General Public License v3.0',
      install_requires=['cvxopt', 'numpy', 'scipy', 'torch'],
      )     
