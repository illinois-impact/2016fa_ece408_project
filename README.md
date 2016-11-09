# ECE 408 Project

The goal of this project is to accelerate the forward propagation of the Convolutional Neural Network(CNN) with GPUs. The sequential implementation provided follows the basic algorithm 16.4 and 16.5 decribed in [book chapter 16](https://wiki.illinois.edu/wiki/display/ece408f16/Book+Chapters?preview=/602518692/603851747/3rd-Edition-Chapter16-case-study-DNN-FINAL.pdf). The dataset and model are from the [MNIST database](http://yann.lecun.com/exdb/mnist/).

## CNN and MNIST

Read the book chapter and familiar youself with CNN.

The model is trained using 60,000 examples(training set images) and the provided data is 10,000 batched queries(test set images). The CNN is expected to get ~97% accuracy on the provided data.

The data and model are in [HDF5](https://support.hdfgroup.org/HDF5/) format.

## CUDA Implementation

Book chapter 16.3 and 16.4 provide a basic CUDA implementation of forward propagation of convolutional layer and possible optimization. You CUDA implementation would be evaluated based on performnace and accuracy. Apply any optimization you think would bring benefit. Also Your CUDA implentation can use any BLAS library(e.g. cuBLAS) but no direct CNN library like cuDNN is allowed.

## How to Build

There are two options to build this project, the first is 
using the [hunter package manager](https://github.com/ruslo/hunter)
the other is using [docker](https://www.docker.com/).


### Using Hunter Package Manager


By default, the compilation uses the [hunter](https://github.com/ruslo/hunter).

This method requires that you have the CUDA toolkit installed on your machine.

### Using Docker


Included is a [Docker](http://docker.io/) build file. This file can be used to build and launch a container which contains the teaching kit labs along with all the software required to run them. Using a GPU within Docker is only supported on Linux, and we recommend using [NVIDIA-Docker](https://github.com/NVIDIA/nvidia-docker) to run the Docker image. To build the Docker container do

~~~
docker build . -t ece408project
~~~

Once built, the `ece408project` image would be listed by the `docker images` command. This will compile your project. You can launch the docker image using

~~~
docker run -it ece408project
~~~



## How to Run


~~~
./ece408 ../data/data.hdf5 ../data/model.hdf5 [batch_size]
~~~

an optional `batch_size` can be specified


## Requirements


The project requires a CUDA supported operating system,
C compiler, and the CUDA 8 Toolkit. The CUDA 8 Toolkit can be downloaded
from the [CUDA Download](https://developer.nvidia.com/cuda-downloads) page.
Instructions on how to install the CUDA Toolkit are available in the
[Quick Start page](http://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).
Installation guides and the list of supported C compilers for [Windows](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html),
[Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), and
[OSX](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html) are
also found in the [CUDA Toolkit Documentation Page](http://docs.nvidia.com/cuda/index.html).

Aside from a C compiler and the CUDA 8 Toolkit, [CMake](https://cmake.org/) 3.1 or later is required
to generate build scripts for your target IDE and compiler. The next section describes
the process of compiling and running a lab.
