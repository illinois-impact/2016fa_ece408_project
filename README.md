# ECE 408 Project

The goal of this project is to accelerate the forward propagation of the Convolutional Neural Network(CNN) with GPUs. The sequential implementation provided follows the basic algorithm 16.4 and 16.5 decribed in [book chapter 16](https://wiki.illinois.edu/wiki/display/ece408f16/Book+Chapters?preview=/602518692/603851747/3rd-Edition-Chapter16-case-study-DNN-FINAL.pdf). The dataset and model are from the [MNIST database](http://yann.lecun.com/exdb/mnist/).

## CNN and MNIST

Read the book chapter and familiarize youself with CNN.

The model is trained using 60,000 examples (training set images) and the provided data is 10,000 batched queries (test set images). The CNN is expected to get ~97% accuracy on the provided data.

The data and model are in [HDF5](https://support.hdfgroup.org/HDF5/) format.

## CUDA Implementation

Book chapter 16.3 and 16.4 provide a basic CUDA implementation of forward propagation of convolutional layer and possible optimization. Your CUDA implementation would be evaluated based on performance and accuracy. Apply any optimization you think would bring benefit. You should not use cuBLAS or cuDNN but we encourage you to compare your implementation with those libraries (profiling information of them could be helpful).

## Requirements

The project requires a CUDA-supported operating system,
C compiler, and the CUDA 8 Toolkit. The CUDA 8 Toolkit can be downloaded
from the [CUDA Download](https://developer.nvidia.com/cuda-downloads) page.
Instructions on how to install the CUDA Toolkit are available in the
[Quick Start page](http://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).
Installation guides and the list of supported C compilers for [Windows](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html),
[Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), and
[OSX](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html) are
also found in the [CUDA Toolkit Documentation Page](http://docs.nvidia.com/cuda/index.html).

Aside from a C compiler and the CUDA 8 Toolkit, [CMake](https://cmake.org/) 3.1 or later is required
to generate build scripts for your target IDE and compiler. 

## How to Build

There are two options to build this project, the first is using the [hunter package manager](https://github.com/ruslo/hunter) and the other is using [docker](https://www.docker.com/).
We sugguest using cmake along with hunter.

### Using Hunter Package Manager

By default, the compilation uses the [hunter](https://github.com/ruslo/hunter).
This method requires that you have the CUDA toolkit installed on your machine.

In the project folder, do
~~~
mkdir build;cd build;cmake ..
~~~

If you need another library (no cuBLAS or cuDNN), you need to modify the CMakeLists.txt.

### Using Docker
 
 [![Docker Automated build](https://img.shields.io/docker/automated/jrottenberg/ffmpeg.svg)](https://hub.docker.com/r/webgpu/ece408project/)

Included is a [Docker](http://docker.io/) build file. This file can be used to build and launch a container which contains this project along with all the software required to run it. Using a GPU within Docker is only supported on Linux, and we recommend using [NVIDIA-Docker](https://github.com/NVIDIA/nvidia-docker) to run the Docker image. To build the Docker container, do

~~~
docker build . -t ece408project
~~~
in the projcet foler. 
Once built, the `ece408project` image would be listed by the `docker images` command. This will compile your project. You can launch the docker image using

~~~
docker run -it ece408project
~~~

## How to Run

~~~
./ece408 ../data/data.hdf5 ../data/model.hdf5 batch_size
~~~

the `batch_size` must match the size of the dataset. If `batch_size` is unspecified, the default value is 10000, which is also the size of `data.hdf5`.

## How to test 

Test your implementation with small batch size frist to verify the correctness. You can parse the data.hdf5 into smaller chuncks using your preferred language(e.g. python). 2, 10 and 100 queries are provides in test2.hdf5, test10.hdf5 and test100.hdf5 in the data folder. Maker sure the data file you feed in has the same batch size as the `batch_size` you specify in the command line.

~~~
./ece408 ../data/test10.hdf5 ../data/model.hdf5 10
~~~

## What to deliver

Code and report. Make sure you have a working CUDA implementation before applying any optimizations. Report should focus on optimizations you have tried and analysis of its benefits if any. No paper limit on report but try to be concise and back your discussion with performance analysis.

