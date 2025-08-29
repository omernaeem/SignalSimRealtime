// filepath: /home/fpga/spoofer/SignalSimRealtime/src/kernel.cu
// kernel.cu - CUDA kernels for parallel signal generation

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <math.h>
#include "BasicTypes.h"
#include "PrnGenerate.h"

// Device constants for PRN generation
__constant__ unsigned char d_dataPrn[1023];
__constant__ unsigned char d_pilotPrn[10230];  // L5 pilot PRN length
__constant__ int d_dataLength;
__constant__ int d_pilotLength;
__constant__ int d_prnAttribute;

// Device function for PRN code generation
__device__ cuComplex GetPrnValueGPU(double curChip, double codeStep, 
                                   cuComplex dataSignal, cuComplex pilotSignal) {
    int chipCount = (int)curChip;
    int dataChip, pilotChip;
    cuComplex prnValue = make_cuFloatComplex(0.0f, 0.0f);
    
    int isBoc = d_prnAttribute & PRN_ATTRIBUTE_BOC;
    int isL2C = d_prnAttribute & PRN_ATTRIBUTE_TMD;
    
    // Data PRN contribution
    if (d_dataLength > 0) {
        dataChip = chipCount % d_dataLength;
        if (isBoc || isL2C)
            dataChip >>= 1;
        
        if (d_dataPrn[dataChip])
            prnValue = cuCsubf(make_cuFloatComplex(0.0f, 0.0f), dataSignal);
        else
            prnValue = dataSignal;
    }
    
    // Pilot PRN contribution
    if (d_pilotLength > 0) {
        pilotChip = chipCount % d_pilotLength;
        if (isBoc || isL2C)
            pilotChip >>= 1;
        
        cuComplex pilotContrib;
        if (d_pilotPrn[pilotChip])
            pilotContrib = cuCsubf(make_cuFloatComplex(0.0f, 0.0f), pilotSignal);
        else
            pilotContrib = pilotSignal;
            
        if (isL2C) {
            if (chipCount & 1) // L2CL slot
                prnValue = pilotContrib;
        } else {
            prnValue = cuCaddf(prnValue, pilotContrib);
        }
    }
    
    // BOC modulation
    if (isBoc && (chipCount & 1))
        prnValue = cuCsubf(make_cuFloatComplex(0.0f, 0.0f), prnValue);
    
    return prnValue;
}

// Device function for phase rotation using fast math
__device__ cuComplex GetRotateValueGPU(unsigned int curIntPhase, int phaseStep) {
    // Convert 32-bit phase to float (0-2π range)
    float angle = __uint2float_rn(curIntPhase) * (2.0f * M_PI) / 4294967296.0f;
    
    // Use fast math functions for better performance
    float cosVal, sinVal;
    __sincosf(angle, &sinVal, &cosVal);
    
    return make_cuFloatComplex(cosVal, sinVal);
}

// CUDA kernel for parallel sample generation
__global__ void generateSamplesKernel(cuComplex* sampleArray, 
                                     double* curChipArray,
                                     double* codeStepArray,
                                     unsigned int* curIntPhaseArray,
                                     int* intPhaseStepArray,
                                     float* ampArray,
                                     cuComplex* dataSignalArray,
                                     cuComplex* pilotSignalArray,
                                     int numMilliseconds,
                                     int samplesPerMs) {
    
    // Calculate global sample index
    int globalSampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSamples = numMilliseconds * samplesPerMs;
    
    if (globalSampleIdx >= totalSamples) return;
    
    // Determine which millisecond this sample belongs to
    int msIdx = globalSampleIdx / samplesPerMs;
    int sampleInMs = globalSampleIdx % samplesPerMs;
    
    // Get parameters for this millisecond
    double baseChip = curChipArray[msIdx];
    double codeStep = codeStepArray[msIdx];
    unsigned int baseIntPhase = curIntPhaseArray[msIdx];
    int intPhaseStep = intPhaseStepArray[msIdx];
    float amplitude = ampArray[msIdx];
    cuComplex dataSignal = dataSignalArray[msIdx];
    cuComplex pilotSignal = pilotSignalArray[msIdx];
    
    // Calculate current chip and phase for this sample
    double curChip = baseChip + sampleInMs * codeStep;
    unsigned int curIntPhase = baseIntPhase + sampleInMs * intPhaseStep;
    
    // Generate PRN value and rotation
    cuComplex prnValue = GetPrnValueGPU(curChip, codeStep, dataSignal, pilotSignal);
    cuComplex rotateValue = GetRotateValueGPU(curIntPhase, intPhaseStep);
    
    // Combine and store result
    sampleArray[globalSampleIdx] = cuCmulf(cuCmulf(prnValue, rotateValue), 
                                          make_cuFloatComplex(amplitude, 0.0f));
}

// CUDA kernel for optimized single millisecond generation (Phase 3 optimization)
__global__ void generateSingleMsKernel(cuComplex* sampleArray,
                                       double curChip,
                                       double codeStep,
                                       unsigned int curIntPhase,
                                       int intPhaseStep,
                                       float amplitude,
                                       cuComplex dataSignal,
                                       cuComplex pilotSignal,
                                       int samplesPerMs) {
    
    int sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (sampleIdx >= samplesPerMs) return;
    
    // Calculate current chip and phase for this sample
    double thisSampleChip = curChip + sampleIdx * codeStep;
    unsigned int thisSampleIntPhase = curIntPhase + sampleIdx * intPhaseStep;
    
    // Generate PRN value and rotation
    cuComplex prnValue = GetPrnValueGPU(thisSampleChip, codeStep, dataSignal, pilotSignal);
    cuComplex rotateValue = GetRotateValueGPU(thisSampleIntPhase, intPhaseStep);
    
    // Combine and store result
    sampleArray[sampleIdx] = cuCmulf(cuCmulf(prnValue, rotateValue), 
                                    make_cuFloatComplex(amplitude, 0.0f));
}

// Host function to launch the kernel
extern "C" {
    cudaError_t launchGenerateSamplesKernel(cuComplex* d_sampleArray,
                                           double* d_curChipArray,
                                           double* d_codeStepArray,
                                           unsigned int* d_curIntPhaseArray,
                                           int* d_intPhaseStepArray,
                                           float* d_ampArray,
                                           cuComplex* d_dataSignalArray,
                                           cuComplex* d_pilotSignalArray,
                                           int numMilliseconds,
                                           int samplesPerMs,
                                           cudaStream_t stream = 0) {
        
        int totalSamples = numMilliseconds * samplesPerMs;
        
        // Calculate optimal block size and grid size
        int blockSize = 256;  // Good balance for most GPUs
        int gridSize = (totalSamples + blockSize - 1) / blockSize;
        
        // Launch kernel
        generateSamplesKernel<<<gridSize, blockSize, 0, stream>>>(
            d_sampleArray,
            d_curChipArray,
            d_codeStepArray,
            d_curIntPhaseArray,
            d_intPhaseStepArray,
            d_ampArray,
            d_dataSignalArray,
            d_pilotSignalArray,
            numMilliseconds,
            samplesPerMs
        );
        
        return cudaGetLastError();
    }
    
    cudaError_t launchSingleMsKernel(cuComplex* d_sampleArray,
                                    double curChip,
                                    double codeStep,
                                    unsigned int curIntPhase,
                                    int intPhaseStep,
                                    float amplitude,
                                    cuComplex dataSignal,
                                    cuComplex pilotSignal,
                                    int samplesPerMs,
                                    cudaStream_t stream = 0) {
        
        // Calculate optimal block size and grid size
        int blockSize = 256;
        int gridSize = (samplesPerMs + blockSize - 1) / blockSize;
        
        // Launch kernel
        generateSingleMsKernel<<<gridSize, blockSize, 0, stream>>>(
            d_sampleArray,
            curChip,
            codeStep,
            curIntPhase,
            intPhaseStep,
            amplitude,
            dataSignal,
            pilotSignal,
            samplesPerMs
        );
        
        return cudaGetLastError();
    }
    
    // Function to copy PRN data to constant memory
    cudaError_t copyPrnToConstantMemory(const unsigned char* h_dataPrn, int dataLength,
                                       const unsigned char* h_pilotPrn, int pilotLength,
                                       int prnAttribute) {
        cudaError_t err;
        
        // Copy data PRN
        if (h_dataPrn && dataLength > 0) {
            err = cudaMemcpyToSymbol(d_dataPrn, h_dataPrn, dataLength * sizeof(unsigned char));
            if (err != cudaSuccess) return err;
        }
        
        // Copy pilot PRN
        if (h_pilotPrn && pilotLength > 0) {
            err = cudaMemcpyToSymbol(d_pilotPrn, h_pilotPrn, pilotLength * sizeof(unsigned char));
            if (err != cudaSuccess) return err;
        }
        
        // Copy lengths and attributes
        err = cudaMemcpyToSymbol(d_dataLength, &dataLength, sizeof(int));
        if (err != cudaSuccess) return err;
        
        err = cudaMemcpyToSymbol(d_pilotLength, &pilotLength, sizeof(int));
        if (err != cudaSuccess) return err;
        
        err = cudaMemcpyToSymbol(d_prnAttribute, &prnAttribute, sizeof(int));
        if (err != cudaSuccess) return err;
        
        return cudaSuccess;
    }
}