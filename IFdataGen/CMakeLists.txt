cmake_minimum_required(VERSION 3.16)

# ============================================================================
# Project
# ============================================================================

# Add CUDA as a project language
project(IFdataGen LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS  OFF)

# CUDA standard (works from CMake 3.12+)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable separable compilation for CUDA translation units
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Optional: let users pass CUDA archs like: -DCMAKE_CUDA_ARCHITECTURES=75;86
# If on CMake < 3.18, this property is ignored; users can still set NVCC flags.
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    # Set a sensible default only if user didn't specify
    if (NOT CMAKE_CUDA_ARCHITECTURES)
        # Common: Turing(75), Ampere(86); adjust to your GPUs
        set(CMAKE_CUDA_ARCHITECTURES 75 86)
    endif()
endif()

# Optional: tune for the build-host CPU
option(USE_NATIVE_OPT "Tune code for the host CPU (-march=native or /arch:AVX2)" ON)

# ============================================================================
# Dependencies
# ============================================================================

find_package(OpenMP QUIET)

# CUDA runtime (optional; CMake can link cudart implicitly, but this target helps)
# Will be found for CUDA-enabled toolchains; if not, we just skip explicit linkage.
find_package(CUDAToolkit QUIET)

# ============================================================================
# Sources and includes
# ============================================================================

# Add your CUDA sources; adjust patterns/paths as needed
file(GLOB SRC CONFIGURE_DEPENDS
     "IFdataGen.cpp"
     "../src/*.cpp"
     "../src/*.cu"          # <-- pick up CUDA kernels
)

add_executable(IFdataGen ${SRC})
target_include_directories(IFdataGen PRIVATE ../inc)

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

    target_compile_options(IFdataGen PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:${MSVC_RELEASE_FLAGS}>
        $<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:C,CXX>>:${MSVC_RELDBG_FLAGS}>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:C,CXX>>:/Od /Zi>)

    # Reasonable NVCC flags under MSVC
    target_compile_options(IFdataGen PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:/W3 /Zc:__cplusplus>)

    # ----- MSVC link options -----
    if (IPO_OK)
        set(MSVC_LINK_LTCG /LTCG /INCREMENTAL:NO)
    else()
        set(MSVC_LINK_LTCG "")
    endif()

    target_link_options(IFdataGen PRIVATE
        $<$<CONFIG:Release>:${MSVC_LINK_LTCG}>
        $<$<CONFIG:RelWithDebInfo>:${MSVC_LINK_LTCG}>)

    # Optional host-CPU tuning (C/C++)
    if (USE_NATIVE_OPT)
        target_compile_options(IFdataGen PRIVATE
            $<$<COMPILE_LANGUAGE:C,CXX>:/arch:AVX2>)
    endif()

else()  # GCC / Clang branch

    # ----- GCC / Clang release flags -----
    set(GCC_CLANG_RELEASE_FLAGS
        -O3 -ffast-math -funroll-loops -fstrict-aliasing)
    if (IPO_OK)
        list(APPEND GCC_CLANG_RELEASE_FLAGS -flto=auto)
    endif()

    target_compile_options(IFdataGen PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:${GCC_CLANG_RELEASE_FLAGS}>
        $<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:C,CXX>>:-O3 -g>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:C,CXX>>:-O0 -g>)

    # CUDA (NVCC) flags on GCC/Clang hosts
    target_compile_options(IFdataGen PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3 --use_fast_math>
        $<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-O3 -lineinfo>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-O0 -G -lineinfo>)

    # Enable LTO at link time if supported (applies to C/C++; NVCC ignores it)
    if (IPO_OK)
        set_property(TARGET IFdataGen PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()

    # Optional host-CPU tuning (C/C++)
    if (USE_NATIVE_OPT)
        target_compile_options(IFdataGen PRIVATE
            $<$<COMPILE_LANGUAGE:C,CXX>:-march=native -mtune=native>)
    endif()

endif()

# ============================================================================
# Link CUDA runtime explicitly if available
# ============================================================================

if (CUDAToolkit_FOUND)
    target_link_libraries(IFdataGen PRIVATE CUDA::cudart)
endif()

# ============================================================================
# OpenMP
# ============================================================================

if (OpenMP_CXX_FOUND)
    target_link_libraries(IFdataGen PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(IFdataGen PRIVATE USE_OPENMP)
endif()

# ============================================================================
# Hot-reload for MSVC (CMP0141)
# ============================================================================

if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
        "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,\
        $<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,\
        $<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

# ============================================================================
# Status output
# ============================================================================

message(STATUS "OpenMP found           : ${OpenMP_CXX_FOUND}")
message(STATUS "IPO / LTO supported    : ${IPO_OK}")
message(STATUS "Host-CPU tuning enabled: ${USE_NATIVE_OPT}")
message(STATUS "CUDA enabled           : ${CMAKE_CUDA_COMPILER}")
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    message(STATUS "CUDA architectures     : ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# TODO: Add tests and install targets if needed.
