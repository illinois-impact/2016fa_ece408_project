
cmake_minimum_required(VERSION 3.0)

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

project(ECE408Project CXX)


include(CMakeToolsHelpers OPTIONAL)

if (UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif (UNIX AND NOT APPLE)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})
set(TOP_DIR "${PROJECT_SOURCE_DIR}")
set(SRC_DIR "${TOP_DIR}/src")

message(STATUS "CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")

set(CONFIG_USE_SYSTEM_HDF5 "OFF" CACHE BOOL "Turn on to enable using the system HDF5 package")

# now hunter can be used, like
hunter_add_package(Sugar)
hunter_add_package(fmt)
if (NOT CONFIG_USE_SYSTEM_HDF5)
  hunter_add_package(ZLIB)
  hunter_add_package(hdf5)
  hunter_add_package(szip)
endif (CONFIG_USE_SYSTEM_HDF5)

### Find dependencies ###
Find_Package(EnableCxx11 REQUIRED)
Find_Package(ZLIB REQUIRED CONFIG)
Find_Package(szip REQUIRED CONFIG)
Find_Package(hdf5 REQUIRED CONFIG)
find_package(fmt REQUIRED CONFIG)
Find_Package(Range REQUIRED)

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

set(PROJECT_TARGET ece408)
add_executable(${PROJECT_TARGET} ${SOURCES})
add_dependencies(${PROJECT_TARGET} range)
target_include_directories(${PROJECT_TARGET}
    PRIVATE "${RANGE_INCLUDE_DIR}"
    PRIVATE "${ZLIB_INCLUDE_DIRS}"
)
target_link_libraries(${PROJECT_TARGET}
    PRIVATE hdf5
    PRIVATE fmt
    PRIVATE ${ZLIB_LIBRARIES}
)

# sugar_status_print("${RANGE_INCLUDE_DIR}")
