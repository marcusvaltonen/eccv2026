# ECCV2026
Official repository for the ECCV 2026 spotlight paper: "Gravity-aware partially
calibrated absolute pose estimation from affine- or rotation-covariant features" by
Marcus Valtonen Örnhag and Alberto Jaenal.

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
More information comming soon.

## Citation
If using our work, please cite:

```
@InProceedings{valtonen-ornhag-jaenal-2026-eccv,
    author    = {Valtonen~{\"O}rnhag, Marcus and Jaenal, Alberto},
    title     = {Gravity-aware partially calibrated absolute pose estimation
from affine- or rotation-covariant features},
    booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
    year      = {2026},
}
```

