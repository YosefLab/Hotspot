import os
from setuptools import setup

# Parse the version string
__version__ = ""
this_directory = os.path.dirname(os.path.abspath(__file__))
version_file = os.path.join(this_directory, "src_python",
                            "hotspot", "_version.py")
exec(open(version_file).read())  # Loads version into __version__

setup(
    name="hotspot",
    version=__version__,
    packages=['hotspot'],
    package_dir={'hotspot': 'src_python/hotspot'},
    include_package_data=True,

    install_requires=[
        'numpy>=1.12',
        'pandas>=0.20',
        'scipy',
        'scikit-learn'],

    author="David DeTomaso",
    author_email="david.detomaso@berkeley.edu",
    description="Getis-ord coefficient",
    keywords="",
    url=""
)
