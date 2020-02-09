from setuptools import setup, find_packages
import os

# Parse version string
this_directory = os.path.dirname(os.path.abspath(__file__))
version_file = os.path.join(this_directory, "hotspot", "_version.py")
exec(open(version_file).read())


setup(
    name="hotspot",
    version=__version__,
    packages=find_packages(),

    install_requires=[
        'matplotlib>=3.0.0',
        'numba>=0.43.1',
        'numpy>=1.16.4',
        'seaborn>=0.9.0',
        'scipy>=1.2.1',
        'pandas>=0.24.0',
        'tqdm>=4.32.2',
        'statsmodels>=0.9.0',
        'scikit-learn>=0.21.2',
    ],
    extras_require=dict(
        test=['pytest>=5.0.0'],
    ),

    include_package_data=True,

    author="David DeTomaso",
    author_email="David.DeTomaso@berkeley.edu",
    description="",
    keywords="",
    url="",
    license=""
)
