cmake_minimum_required(VERSION 3.18)

# ============================================================================
# Project
# ============================================================================

project(CudaTest LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS  OFF)

# Optional: tune for the build-host CPU
option(USE_NATIVE_OPT "Tune code for the host CPU (-march=native or /arch:AVX2)" ON)

# Tell CMake how to treat CUDA arch detection & avoid empty arch warnings
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

# Default GPU architectures for all CUDA targets in this dir
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# ============================================================================
# Dependencies
# ============================================================================

find_package(OpenMP QUIET)
find_package(CUDAToolkit REQUIRED)

# ============================================================================
# Sources and includes
# ============================================================================

file(GLOB SRC CONFIGURE_DEPENDS
     "CudaTest.cpp"
     "../src/*.cpp")

# Exclude the IFdataGen.cpp to avoid main function conflicts
list(FILTER SRC EXCLUDE REGEX ".*IFdataGen\\.cpp$")

add_executable(CudaTest ${SRC})

# Create second executable for advanced guide
file(GLOB ADVANCED_SRC CONFIGURE_DEPENDS
     "AdvancedCudaGuide.cpp"
     "../src/*.cpp")

# Exclude the IFdataGen.cpp to avoid main function conflicts
list(FILTER ADVANCED_SRC EXCLUDE REGEX ".*IFdataGen\\.cpp$")

add_executable(AdvancedCudaGuide ${ADVANCED_SRC})

# Create third executable for GetIfSample data test
file(GLOB DATA_TEST_SRC CONFIGURE_DEPENDS
     "TestGetIfSampleData.cpp"
     "../src/*.cpp")

# Exclude the IFdataGen.cpp to avoid main function conflicts
list(FILTER DATA_TEST_SRC EXCLUDE REGEX ".*IFdataGen\\.cpp$")

add_executable(TestGetIfSampleData ${DATA_TEST_SRC})

# Create fourth executable for signal comparison
file(GLOB COMPARE_SRC CONFIGURE_DEPENDS
     "CompareSignals.cpp"
     "../src/*.cpp")

# Exclude the IFdataGen.cpp to avoid main function conflicts
list(FILTER COMPARE_SRC EXCLUDE REGEX ".*IFdataGen\\.cpp$")

add_executable(CompareSignals ${COMPARE_SRC})

target_include_directories(CudaTest PRIVATE ../inc)
target_include_directories(AdvancedCudaGuide PRIVATE ../inc)
target_include_directories(TestGetIfSampleData PRIVATE ../inc)
target_include_directories(CompareSignals PRIVATE ../inc)

# Add CUDA source files and link CUDA libraries
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../src/kernel.cu")
    target_sources(CudaTest PRIVATE ../src/kernel.cu)
    target_sources(AdvancedCudaGuide PRIVATE ../src/kernel.cu)
    target_sources(TestGetIfSampleData PRIVATE ../src/kernel.cu)
    target_sources(CompareSignals PRIVATE ../src/kernel.cu)
endif()
target_link_libraries(CudaTest PRIVATE CUDA::cudart CUDA::cuda_driver)
target_link_libraries(AdvancedCudaGuide PRIVATE CUDA::cudart CUDA::cuda_driver)
target_link_libraries(TestGetIfSampleData PRIVATE CUDA::cudart CUDA::cuda_driver)
target_link_libraries(CompareSignals PRIVATE CUDA::cudart CUDA::cuda_driver)
target_include_directories(CudaTest PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(AdvancedCudaGuide PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(TestGetIfSampleData PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(CompareSignals PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(CudaTest PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
set_target_properties(AdvancedCudaGuide PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
set_target_properties(TestGetIfSampleData PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
set_target_properties(CompareSignals PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# IPO / LTO support check
# ============================================================================

include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_OK OUTPUT IPO_ERR)

# ============================================================================
# Compiler and linker flags
# ============================================================================

if (MSVC)

    # ----- MSVC release and relwithdebinfo flag lists -----
    set(MSVC_RELEASE_FLAGS
        /O2 /Ot /fp:fast /Ob3 /Zi /permissive-)
    if (IPO_OK)
        list(APPEND MSVC_RELEASE_FLAGS /GL)
    endif()

    set(MSVC_RELDBG_FLAGS
        /O2 /Zi)
    if (IPO_OK)
        list(APPEND MSVC_RELDBG_FLAGS /GL)
    endif()

    target_compile_options(CudaTest PRIVATE
        $<$<CONFIG:Release>:${MSVC_RELEASE_FLAGS}>
        $<$<CONFIG:RelWithDebInfo>:${MSVC_RELDBG_FLAGS}>
        $<$<CONFIG:Debug>:/Od /Zi>)
    
    target_compile_options(AdvancedCudaGuide PRIVATE
        $<$<CONFIG:Release>:${MSVC_RELEASE_FLAGS}>
        $<$<CONFIG:RelWithDebInfo>:${MSVC_RELDBG_FLAGS}>
        $<$<CONFIG:Debug>:/Od /Zi>)

    target_compile_options(TestGetIfSampleData PRIVATE
        $<$<CONFIG:Release>:${MSVC_RELEASE_FLAGS}>
        $<$<CONFIG:RelWithDebInfo>:${MSVC_RELDBG_FLAGS}>
        $<$<CONFIG:Debug>:/Od /Zi>)

    target_compile_options(DiagnosticTest PRIVATE /arch:AVX2 /O2)

    # ----- MSVC link options -----
    if (IPO_OK)
        set(MSVC_LINK_LTCG /LTCG /INCREMENTAL:NO)
    else()
        set(MSVC_LINK_LTCG "")
    endif()

    target_link_options(CudaTest PRIVATE
        $<$<CONFIG:Release>:${MSVC_LINK_LTCG}>
        $<$<CONFIG:RelWithDebInfo>:${MSVC_LINK_LTCG}>)
    
    target_link_options(AdvancedCudaGuide PRIVATE
        $<$<CONFIG:Release>:${MSVC_LINK_LTCG}>
        $<$<CONFIG:RelWithDebInfo>:${MSVC_LINK_LTCG}>)

    # Optional host-CPU tuning
    if (USE_NATIVE_OPT)
        target_compile_options(CudaTest PRIVATE /arch:AVX2)
        target_compile_options(AdvancedCudaGuide PRIVATE /arch:AVX2)
    endif()

else()  # GCC / Clang branch

    # ----- GCC / Clang release flags -----
    set(GCC_CLANG_RELEASE_FLAGS
        -O3 -ffast-math -funroll-loops -fstrict-aliasing)
    if (IPO_OK)
        list(APPEND GCC_CLANG_RELEASE_FLAGS -flto=auto)
    endif()

    target_compile_options(CudaTest PRIVATE
        $<$<CONFIG:Release>:${GCC_CLANG_RELEASE_FLAGS}>
        $<$<CONFIG:RelWithDebInfo>:-O3 -g>
        $<$<CONFIG:Debug>:-O0 -g>)
    
    target_compile_options(AdvancedCudaGuide PRIVATE
        $<$<CONFIG:Release>:${GCC_CLANG_RELEASE_FLAGS}>
        $<$<CONFIG:RelWithDebInfo>:-O3 -g>
        $<$<CONFIG:Debug>:-O0 -g>)

    # Enable LTO at link time if supported
    if (IPO_OK)
        set_property(TARGET CudaTest PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        set_property(TARGET AdvancedCudaGuide PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()

    # Optional host-CPU tuning
    if (USE_NATIVE_OPT)
        target_compile_options(CudaTest PRIVATE -march=native -mtune=native)
        target_compile_options(AdvancedCudaGuide PRIVATE -march=native -mtune=native)
    endif()

endif()

# ============================================================================
# OpenMP
# ============================================================================

if (OpenMP_CXX_FOUND)
    target_link_libraries(CudaTest PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(AdvancedCudaGuide PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(CudaTest PRIVATE USE_OPENMP)
    target_compile_definitions(AdvancedCudaGuide PRIVATE USE_OPENMP)
endif()

# ============================================================================
# Status output
# ============================================================================

message(STATUS "OpenMP found           : ${OpenMP_CXX_FOUND}")
message(STATUS "IPO / LTO supported    : ${IPO_OK}")
message(STATUS "Host-CPU tuning enabled: ${USE_NATIVE_OPT}")
