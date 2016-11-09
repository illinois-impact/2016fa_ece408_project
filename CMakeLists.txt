
cmake_minimum_required(VERSION 3.0)

cmake_policy(SET CMP0007 OLD)
cmake_policy(SET CMP0054 OLD)
if (POLICY CMP0025)
    cmake_policy(SET CMP0025 OLD) # report Apple's Clang as just Clang
endif (POLICY CMP0025)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


include("cmake/modules/HunterGate.cmake")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.16.24.tar.gz"
    SHA1 "00ab627d3d4fe5151e15b6ab2b1b1da17650984a"
)

project(ECE408Project)


include(CMakeToolsHelpers OPTIONAL)

if (UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif (UNIX AND NOT APPLE)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})
set(TOP_DIR "${PROJECT_SOURCE_DIR}")
set(SRC_DIR "${TOP_DIR}/src")

set(CONFIG_USE_SYSTEM_HDF5 "OFF" CACHE BOOL "Turn on to enable using the system HDF5 package")

# now hunter can be used, like
hunter_add_package(Sugar)
if (NOT CONFIG_USE_SYSTEM_HDF5)
  # hunter_add_package(ZLIB)
  hunter_add_package(hdf5)
  # hunter_add_package(szip)
endif (NOT CONFIG_USE_SYSTEM_HDF5)

### Find dependencies ###
Find_Package(EnableCxx11 REQUIRED)
Find_Package(ZLIB CONFIG REQUIRED)
Find_Package(szip CONFIG REQUIRED)
Find_Package(hdf5 CONFIG REQUIRED)
Find_Package(Range REQUIRED)
Find_Package(CUDA REQUIRED)

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CUDA_VERBOSE_BUILD OFF)

### Configure sugar ###

include("${SUGAR_ROOT}/cmake/Sugar")
include(sugar_status_print)
include(sugar_groups_generate)
include(sugar_include)
include(sugar_generate_warning_flags)


### Add source files ###
sugar_include("${SRC_DIR}")
sugar_groups_generate(${SOURCES})

### Setup target ###

set(CUDA_PROPAGATE_HOST_FLAGS TRUE)

# It may be necessary to specify the compute capability of your
# NVIDIA GPU. In that case, simply uncomment the following two lines that start
# with 'set' and change the value of COMPUTE_CAPABILITY to one of the following
# hardware architectures: Tesla-class '11', '12', or '13'; Fermi-class '20'
# or '21'; Kepler-class '30', '32', '35'; or Maxwell-class '50' or '52'
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.0 3.5+PTX 5.0 Maxwell 6.1)
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

# Set warning levels for host compilation
if (UNIX)
    add_compile_options(-Wextra)
    add_compile_options(-Wno-unused-function)
    add_compile_options(-Wno-unused-parameter)
    add_compile_options(-Wno-unused-parameter)
    add_compile_options(-Wno-vla)
    add_compile_options(-Wno-unused-variable)
endif (UNIX)

# Add debugging to CUDA NVCC flags (for NVidia's NSight tools)
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

################################
################################

include_directories(
    "${RANGE_INCLUDE_DIR}"
    "${ZLIB_INCLUDE_DIRS}"
    "${HDF5_INCLUDE_DIR}"
    "${SRC_DIR}"
)

sugar_status_print("HDF5_INCLUDE_DIRS = ${HDF5_INCLUDE_DIR}")

set(PROJECT_TARGET ece408)
cuda_add_executable(${PROJECT_TARGET} ${SOURCES})
add_dependencies(${PROJECT_TARGET} range)
target_link_libraries(${PROJECT_TARGET}
    hdf5
    ${ZLIB_LIBRARIES}
)

# sugar_status_print("${RANGE_INCLUDE_DIR}")

# Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()
