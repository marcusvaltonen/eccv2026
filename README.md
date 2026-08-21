<div align="center">
<h1>Gravity-aware partially calibrated absolute <br> pose estimation from affine- or <br> rotation-covariant features</h1>
<a href="_blank"><img src="https://img.shields.io/badge/Paper-blue" alt="Paper"></a>
<a href="https://marcusvaltonen.github.io/eccv2026/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<br>
<br>
<strong>
<a href="https://scholar.google.com/citations?user=U_U-GmcAAAAJ">Marcus Valtonen Örnhag<sup>1</sup></a>
&nbsp;&nbsp;
<a href="https://scholar.google.com/citations?user=UYL0UY0AAAAJ">Alberto Jaenal<sup>2</sup></a>
&nbsp;&nbsp;
<a href="https://scholar.google.com/citations?user=jLJ6ZaMAAAAJ">Stefan Adalbjörnsson<sup>1</sup></a>
<br>
<br>
<sup>1</sup> Ericsson Research &nbsp;&nbsp;
<sup>2</sup> University of Zaragoza
</strong>
<br>
<br>
<br>
</div>

Official repository for the ECCV 2026 spotlight paper.

## Installation
This project uses CMake, which needs to be available to compile it

For Ubuntu

```console
sudo apt-get install cmake libeigen3-dev
```

Furthermore, [PoseLib](https://github.com/PoseLib/PoseLib/tree/master) is required. Follow the installation
instruction in the repo.

Then you may use the build script

```console
./build.sh [-t]
```

This generates an executable in the ``_build`` folder which you can execute.
Use the flag `-t` to build the tests.

## Running the example and test scripts

To run the example script, run

```console
 ./_build/examples/absolute_pose/test_random_problems 0 0 0 0 0 1
```

where the six arguments expected are ``[point_noise] [angle_noise] [affine_noise] [normal_noise] [imu_noise] [nprob]``.

Run the tests (if compiled) the following way

```console
$ cd _build && ctest -j $nproc --output-on-failure
```


## GC-RANSAC integration and python bindings
This code has been integrated in GC-RANSAC, see this [repo](https://github.com/AlbertoJaenal/graph-cut-ransac/tree/eccv26).
It includes instructions on how to compile and run it with python bindings.

## Citation
If using our work, please cite:

```
@InProceedings{valtonen-ornhag-jaenal-2026-eccv,
    author    = {Valtonen~{\"O}rnhag, Marcus and Jaenal, Alberto and Adalbj{\"o}rnsson, Stefan},
    title     = {Gravity-aware partially calibrated absolute pose estimation
from affine- or rotation-covariant features},
    booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
    year      = {2026},
}
```

