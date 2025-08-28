cmake_minimum_required(VERSION 3.18)  # Need >=3.18 for full CUDA support
project(MyProject LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Optional: specify architecture (adjust as needed for your GPU)
# set(CMAKE_CUDA_ARCHITECTURES 75)  # Example: Turing

# Add subdirectory for IFdataGen
add_subdirectory(IFdataGen)

