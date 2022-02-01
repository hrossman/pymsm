import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='pymsm',
    version='0.0.2',
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    description='Python implementation of a Multi-state competing risk model',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Hagai Rossman, Ayya Keshet',
    author_email='hagairossman@gmail.com',
    url='https://github.com/hrossman/pymsm',
    project_urls={'Documentation': 'https://pymsm.readthedocs.io/'},
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'lifelines',
        'scikit-learn'
    ],
)
