cmake_minimum_required(VERSION 3.18)  # 3.18+ recommended for CUDA arch handling
project(1_exercise LANGUAGES CXX)

# Detect CUDA support first
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(HAS_CUDA TRUE)
else()
  set(HAS_CUDA FALSE)
endif()

# Use the modern policy so empty archs are not allowed silently
cmake_policy(SET CMP0104 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----- Sources -----
if(HAS_CUDA)
  set(CUDA_SOURCES ${CMAKE_SOURCE_DIR}/src/kernel_gpu.cu)
endif()

add_executable(${PROJECT_NAME}
  ${CMAKE_SOURCE_DIR}/src/main.cpp
  ${CUDA_SOURCES}
)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ----- CUDA config (only if available) -----
if(HAS_CUDA)
  # If the user didn't pass -DCMAKE_CUDA_ARCHITECTURES=..., set a sensible default
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    # Pick what's right for your machine or CI:
    # set(CMAKE_CUDA_ARCHITECTURES native)        # auto-detect on the build host
    set(CMAKE_CUDA_ARCHITECTURES 80)             # A100
    # Examples:
    # set(CMAKE_CUDA_ARCHITECTURES 86)           # RTX 30-series (Ampere)
    # set(CMAKE_CUDA_ARCHITECTURES 89)           # RTX 40-series (Ada)
    # set(CMAKE_CUDA_ARCHITECTURES 70;75;80)     # multi-arch fat binary
  endif()

  # Apply CUDA properties to the target
  set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  )

  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

  target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xptxas -O3>)

  target_compile_definitions(${PROJECT_NAME} PRIVATE MYLIB_HAS_CUDA)
endif()

# C++ warnings
target_compile_options(${PROJECT_NAME} PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)