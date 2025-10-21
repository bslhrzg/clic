cmake_minimum_required(VERSION 3.18)
# Project name is for CMake internal use, this is a good descriptive name
project(clic_backend LANGUAGES CXX)

# Options
option(BUILD_WITH_OPENMP "Enable OpenMP parallelism" ON)
option(BUILD_WITH_MPI    "Enable MPI build"          OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# pybind11
find_package(pybind11 CONFIG REQUIRED)

# OpenMP
if(BUILD_WITH_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

# MPI (optional)
if(BUILD_WITH_MPI)
  find_package(MPI REQUIRED)
  add_compile_definitions(CI_USE_MPI)
endif()

# Sources - CORRECTED PATHS
set(CORE_SOURCES
  clic/cpp_src/ci_core.cpp
  clic/cpp_src/slater_condon.cpp
  clic/cpp_src/hamiltonian.cpp
  clic/cpp_src/ed_tools.cpp
  clic/cpp_src/bindings.cpp
)

# Extension module - No change to the name here.
# This correctly defines the C++ module name as 'clic_clib'
pybind11_add_module(clic_clib ${CORE_SOURCES})

# Tell CMake to put the compiled library inside the 'clic' package directory.
set_target_properties(clic_clib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/clic"
)

# CORRECTED PATH
target_include_directories(clic_clib PRIVATE clic/cpp_src)

if(BUILD_WITH_OPENMP)
  target_link_libraries(clic_clib PRIVATE OpenMP::OpenMP_CXX)
endif()

if(BUILD_WITH_MPI)
  target_link_libraries(clic_clib PRIVATE MPI::MPI_CXX)
endif()

# Faster numeric builds
if (MSVC)
  target_compile_options(clic_clib PRIVATE /O2 /DNOMINMAX)
else()
  target_compile_options(clic_clib PRIVATE -O3 -march=native -mtune=native)
endif()

# Helpful output
message(STATUS "BUILD_WITH_OPENMP=${BUILD_WITH_OPENMP}")
message(STATUS "BUILD_WITH_MPI=${BUILD_WITH_MPI}")
