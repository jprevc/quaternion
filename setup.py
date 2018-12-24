from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='quaterion',
      version='1.0',
      description='A simple and efficient library for computations with quaternions.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Jost Prevc',
      author_email='jost.prevc@gmail.com',
      url="https://github.com/jprevc/quaternion",
      packages=['quaternion'],
      install_requires=['numpy'],
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
      ],
      )