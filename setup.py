import os
from setuptools import setup

setup(name='MatrixFact',
      version='1.0',
      description='Python Matrix Factorization Module',
      author='Genta Indra Winata',
      author_email='gentaindrawinata@gmail.com',
      url='https://github.com/gentaiscool/matrix_fact',
      packages = ['matrix_fact'],    
      license = 'GNU General Public License v3.0',
      install_requires=['cvxopt', 'numpy', 'scipy'],
      long_description=open('README.md').read(),
      )     
