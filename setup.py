from setuptools import find_packages
from setuptools import setup


URL = 'https://github.com/brandontrabucco/semantic-aug'
DESCRIPTION = "Semantic Controls For Data Augmentation"
CLASSIFIERS = ['Intended Audience :: Developers',
               'Intended Audience :: Science/Research',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Artificial Intelligence',
               'Topic :: Scientific/Engineering :: Mathematics',
               'Topic :: Software Development',
               'Topic :: Software Development :: Libraries',
               'Topic :: Software Development :: Libraries :: Python Modules',
               'License :: OSI Approved :: MIT License',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8',
               'Programming Language :: Python :: 3.9']


with open('README.md', 'r') as readme:
    LONG_DESCRIPTION = readme.read()  # use readme as long description


setup(name='semantic-aug', version='1.0', license='MIT',
      author='Brandon Trabucco', author_email='brandon@btrabucco.com',
      packages=find_packages(include=['semantic_aug', 'semantic_aug.*']),
      classifiers=CLASSIFIERS, description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      url=URL, keywords=['Computer Vision', 'Data Augmentation'],
      install_requires=['torch', 'torchvision', 'pandas'])