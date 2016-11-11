# ECE 408 Project

The goal of this project is to accelerate the forward propagation step of the Convolutional Neural Network (CNN) algorithm using GPUs. The sequential implementation provided follows the basic algorithm 16.4 and 16.5 decribed in [book chapter 16](https://wiki.illinois.edu/wiki/display/ece408f16/Book+Chapters?preview=/602518692/603851747/3rd-Edition-Chapter16-case-study-DNN-FINAL.pdf). The dataset and model are from the [MNIST database](http://yann.lecun.com/exdb/mnist/).

## CNN and MNIST

Read the book chapter and familiarize youself with the CNN algorithm.

Provided is a model that has been trained using 60,000 examples (training set images) and the provided test data is 10,000 batched queries (test set images). The expected accuracy of the CNN is `~97%` on the provided test dataset.

The data and model are in [HDF5](https://support.hdfgroup.org/HDF5/) format and we have provided the code to read the input model and the training dataset.

## CUDA Implementation

Book chapters 16.3 and 16.4 provide a basic CUDA implementation of forward propagation of convolutional layer and possible optimization. Your CUDA implementation would be evaluated based on performance and accuracy. Apply any optimization you think would bring benefit and feel free to modify any part of the code. You should not use `cuBLAS` or `cuDNN` for the implementation, but you are expected to compare your implementation with those libraries --- profiling the code as well as comparing the algorithms used (if algorithm information is publically available).

## Requirements

The project requires a CUDA-supported operating system, C compiler, and the CUDA 8 Toolkit. The CUDA 8 Toolkit can be downloaded from the [CUDA Download](https://developer.nvidia.com/cuda-downloads) page. Instructions on how to install the CUDA Toolkit are available in the [Quick Start page](http://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html). Installation guides and the list of supported C compilers for [Windows](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html), [Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), and [OSX](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html) are also found in the [CUDA Toolkit Documentation Page](http://docs.nvidia.com/cuda/index.html).

Aside from a C compiler and the CUDA 8 Toolkit, [CMake](https://cmake.org/) 3.1 or later is required to generate build scripts for your target IDE and compiler.

## How to Build

There are two options to build this project, the first is using the [Hunter Package Manager](https://github.com/ruslo/hunter) and the other is using [Docker](https://www.docker.com/). We sugguest using CMake along with Hunter, but it is known not to work on all operating systems. In this case, we suggest that you either using Docker or install the libraries needed (mainly `HDF5`).

### Using Hunter Package Manager

By default, the compilation uses the [Hunter](https://github.com/ruslo/hunter) --- a C package manager. This method requires that you have the CUDA toolkit installed on your machine.

Assuming that you have checked out the project into `$SRCDIR` do

~~~
cd $SRCDIR
mkdir build
cd build
cmake $SRCDIR
~~~

This will download the required software needed for the project. You may see some warning while the system is compiling _HDF5_, which you can ignore. Once CMake has been run, a `Makefile` is generated so you can then perform `make` to buidl the project.

~~~
make
~~~

If you do not plan on using `make`, examine the `cmake -G` option which allows you to generate XCode, Visual Studio, ... project configurations. You may also need to change the build type to enable/disable debugging and/or optimizations.

If you need to use another library, you need have to modify the `CMakeLists.txt` and add the libraries to the `target_link_libraries` (and possibly the `include_directories`) section.

### Using Docker

[![Docker Automated build](https://img.shields.io/docker/automated/jrottenberg/ffmpeg.svg)](https://hub.docker.com/r/webgpu/ece408project/)

Also included is a [Docker](http://docker.io/) build file. This file is a specification for a Docker container image. It can be used to build and launch a container (think of a virtual machine) which contains this project along with all the software required to run it. Using a GPU within Docker is only supported on Linux(you can compile and run the serial code on any operating system), and we recommend using [NVIDIA-Docker](https://github.com/NVIDIA/nvidia-docker) to run the Docker image. To build the Docker container, do

~~~
cd $SRCDIR
docker build . -t ece408project
~~~

Once built, the `ece408project` image would be listed by the `docker images` command. This will compile your project. You can launch the docker image using

~~~
docker run -it ece408project
~~~

## How to Run

~~~
./ece408 ../data/testdata.hdf5 ../data/model.hdf5 batch_size
~~~

the `batch_size` must match the size of the dataset. If `batch_size` is unspecified, the default value is 10000, which is also the size of `data.hdf5`.

## How to Test

Test your implementation with small batch size frist to verify the correctness. You can parse the data.hdf5 into smaller chuncks using your preferred language(e.g. python). 2, 10 and 100 queries are provides in test2.hdf5, test10.hdf5 and test100.hdf5 in the data folder. Maker sure the data file you feed in has the same batch size as the `batch_size` you specify in the command line.

~~~
./ece408 ../data/test10.hdf5 ../data/model.hdf5 10
~~~

## What to deliver

Code and report. Make sure you have a working CUDA implementation before applying any optimizations. Report should focus on optimizations you have tried and analysis of its benefits if any. No paper limit on report but try to be concise and back your discussion with performance analysis.


## Utilities

We provide a few utilities in the `utils.hpp` file.

### How to Time

In `utils.hpp` a function called `now()` which allows you to get the current time at a high resolution. To measure the overhead of a function `f(args...)`, the pattern to use is:

~~~{.cpp}
const auto tic = now();
f(args...);
const auto toc = now();
const auto elapsed = std::chrono::duration<double, std::milli>(toc - tic).count();;
std::cout << "Calling f(args...) took " << elapsed << "milliseconds\n";
~~~


### Range Loops

Throughout the serial code, we use the [`range.hpp`](https://github.com/harrism/cpp11-range) to make the code easier to understand. Essentially,


~~~{.cpp}
for (const auto ii : range(0, N)) {
    do_stuff(ii);
}
~~~

Is equivalent to

~~~{.cpp}
for (const auto ii = 0; ii < N; ii++) {
    do_stuff(ii);
}
~~~


## Issues

Please use the [Github issue manager](https://github.com/webgpu/ece408project/issues) to report any issues or suggestions about the project.
>>>>>>> 34506be46acaf1c5aa942df9f16a4c15230912de
