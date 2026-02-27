cmake_minimum_required(VERSION 3.20)
project(mlxbridge LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Disable MLX extras we don't need
set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Add MLX as subdirectory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/mlx mlx_build)

# Collect all bridge source files
file(GLOB BRIDGE_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)
file(GLOB_RECURSE MODEL_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/models/*.cpp
)
list(APPEND BRIDGE_SOURCES ${MODEL_SOURCES})

# Create static library
add_library(mlxbridge STATIC ${BRIDGE_SOURCES})

target_include_directories(mlxbridge PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/mlx
)

target_link_libraries(mlxbridge PUBLIC mlx)
