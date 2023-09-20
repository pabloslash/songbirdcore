from setuptools import setup, find_packages

# install_requires = open_requirements('requirements.txt')

# d = {}
# exec(open("spikefinder/version.py").read(), None, d)
# version = d['version']

long_description = open("README.md").read()

setup(name='songbird-core',
      version='0.1',
      description='Python toolkit for analysis of audio and neural recordings from songbirds.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/pabloslash/songbird-core",
      packages=find_packages(),
      author='Pablo M. Tostado',
      author_email='tostadomarcos@gmail.com',
      license='MIT',
#       install_requires = install_requires,
#       packages=['SpikeFinder'],
      install_requires=['numpy',
                        'matplotlib',
                        'pandas',
                        'librosa'
                       ],
      
      classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
      zip_safe=False)
