![SIMply logo](simply_banner.png)

# SIMply: Python Image Simulator for Planetary Exploration
SIMply is an open-source python tool for simulating physically realistic images.
SIMply is designed primarily to support the development of spaceborne cameras by providing a simple and accessible simulation tool for generating images of space scenes.
SIMply can simulate a wide range of spaceborne imaging scenarios (small bodies, moons, planets, artificial objects), and provides a high level of flexibility to customise the camera, scene and rendering engine.
Simulations can be set up intuitively in just a few lines of Python code.

As a physically-based image simulator, SIMply is also suitable for simulating a wide range of imaging scenarios unrelated to space. Thanks to its simple design, SIMply is also a good way to experiment with and learn about physical image simulation.

## Install
Currently, there are two options for installation:
- clone the SIMply repository
- simply download the SIMply source code (select code > download zip) and place the full SIMply package into your project

### Dependencies
In both cases, it is also necessary to ensure the following dependencies are installed:
- [open3D](https://www.open3d.org/)
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [openCV](https://opencv.org/)
- [astropy](http://www.astropy.org)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [gdal](https://pypi.org/project/GDAL/) (optional, for working with geotifs)

## Software Paper
SIMply has an accompanying peer-reviewed journal [paper](https://doi.org/10.34133/space.0319) describing its capabilities, showcasing example simulations, and providing validation/verification of SIMply's physical accuracy. The paper, published in the journal Space: Science & Technology, is accessible [here](https://doi.org/10.34133/space.0319).

If citing this software, please use the above paper.

SIMply is supporting a number of research [projects](https://spaceborneplanetaryimaging.com/).

## Usage
See the [examples.py]('examples/examples.py) file for simple examples of how to set up and run image simulations. More detailed examples and user guides will be released in the near future.

## Example Outputs
### Comet 67P
![c67p image](examples/example_outputs/c67p.png)
### Moon Colour Image
![RGB Moon image](examples/example_outputs/moon_rgb.jpg)
### Spacecraft Rendezvous
![RPO vid](examples/example_outputs/RPO.gif)

## Contact
Want to work together? Require support integrating SIMply's capabilities into a project? Get in touch [here](https://spaceborneplanetaryimaging.com/contact/) for collaboration and consulting enquiries.

Currently, pull requests will be considered for minor changes/bug fixes only.

## Acknowledgements

[![numpy](http://img.shields.io/badge/powered%20by-NumPy-blue.svg?style=flat)](https://numpy.org/)
[![open3d](http://img.shields.io/badge/powered%20by-Open3D-blue.svg?style=flat)](https://www.open3d.org/)
[![scipy](http://img.shields.io/badge/powered%20by-SciPy-blue.svg?style=flat)](https://scipy.org/)
[![opencv](http://img.shields.io/badge/powered%20by-OpenCV-blue.svg?style=flat)](https://opencv.org/)
[![pandas](http://img.shields.io/badge/powered%20by-pandas-blue.svg?style=flat)](https://pandas.pydata.org/)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-blue.svg?style=flat)](http://www.astropy.org/)
[![matplotlib](http://img.shields.io/badge/powered%20by-Matplotlib-blue.svg?style=flat)](https://matplotlib.org/)
[![gdal](http://img.shields.io/badge/powered%20by-gdal-blue.svg?style=flat)](https://pypi.org/project/GDAL/)

Logo Design: Divya M. Persaud

## Licence
See [LICENCE.txt](LICENCE.txt)