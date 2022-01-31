import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='PyMSM',
    version='0.0.1',
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    description='Python implementation of a Multi-state competing risk model',
    author='Hagai Rossman, Ayya Keshet',
    author_email='hagairossman@gmail.com',
    url='https://github.com/hrossman/PyMSM',
    project_urls={'Documentation': 'https://pymsm.readthedocs.io/'},
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'lifelines',
    ],
)
