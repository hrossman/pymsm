from setuptools import setup

setup(
  name='PyMSM',
  packages=['PyMSM'],
  version='0.1',
  description='Python implemantation of a Multistate competing risk model',
  author='Hagai Rossman, Ayya Keshet',
  author_email='hagairossman@gmail.com',
  url='https://github.com/hrossman/PyMSM',
  #download_url='https://github.com/user/reponame/archive/v_01.tar.gz',    # TODO
  install_requires=[
      'numpy',
      'pandas',
      'scipy',
      'lifelines',
      ],
)
