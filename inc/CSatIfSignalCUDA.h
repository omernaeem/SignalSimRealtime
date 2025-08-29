

// CSatIfSignalCUDA.h
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#include <memory>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <stdexcept>

#include "BasicTypes.h"
#include "PrnGenerate.h"
#include "NavBit.h"
#include "SatelliteSignal.h"

// External CUDA kernel function declarations
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
                                           cudaStream_t stream);
    
    cudaError_t launchSingleMsKernel(cuComplex* d_sampleArray,
                                    double curChip,
                                    double codeStep,
                                    unsigned int curIntPhase,
                                    int intPhaseStep,
                                    float amplitude,
                                    cuComplex dataSignal,
                                    cuComplex pilotSignal,
                                    int samplesPerMs,
                                    cudaStream_t stream);
    
    cudaError_t copyPrnToConstantMemory(const unsigned char* h_dataPrn, int dataLength,
                                       const unsigned char* h_pilotPrn, int pilotLength,
                                       int prnAttribute);
}

// Structure to store parameters for each millisecond
struct MillisecondParams {
    double CurChip;
    double CodeStep;
    unsigned int CurIntPhase;
    int IntPhaseStep;
    double Amp;
    cuComplex DataSignal;
    cuComplex PilotSignal;
};

class CSatIfSignalCUDA {
public:
    CSatIfSignalCUDA(int sampleNumber, int ifFreq, GnssSystem system, int signalIndex, unsigned char svid);
    ~CSatIfSignalCUDA();
    void InitState(GNSS_TIME CurTime, PSATELLITE_PARAM pSatParam, NavBit* pNavData);
	void GetIfSample(GNSS_TIME CurTime);
	void CollectParameters(GNSS_TIME CurTime);  // New method to collect parameters
	void GenerateSamplesFromParams();           // New method to generate samples from collected parameters
	void ClearParameterBuffer();                // Clear the parameter buffer
    cuComplex *SampleArray;

private:
    int         SampleNumber;
    int         IfFreq;
    GnssSystem  System;
    int         SignalIndex;
    int         Svid;

    std::unique_ptr<PrnGenerate> PrnSequence;
	int DataLength, PilotLength;
	CSatelliteSignal SatelliteSignal;
	PSATELLITE_PARAM SatParam;
	double StartCarrierPhase, EndCarrierPhase;
	GNSS_TIME StartTransmitTime, EndTransmitTime, SignalTime;
	cuComplex DataSignal, PilotSignal;
	int GlonassHalfCycle, HalfCycleFlag;

	// Parameter storage for batch processing
	std::vector<MillisecondParams> ParameterBuffer;
	int MaxMilliseconds;

	// GPU memory management
	cuComplex* d_sampleArray;
	double* d_curChipArray;
	double* d_codeStepArray;
	unsigned int* d_curIntPhaseArray;
	int* d_intPhaseStepArray;
	float* d_ampArray;
	cuComplex* d_dataSignalArray;
	cuComplex* d_pilotSignalArray;
	bool gpuMemoryAllocated;
	
	void AllocateGPUMemory();
	void FreeGPUMemory();
	void CopyParametersToGPU();

	cuComplex GetPrnValue(double &CurChip, double CodeStep);
	cuComplex GetRotateValue(double & CurPhase, double PhaseStep);
	cuComplex GetRotateValue(unsigned int & CurPhase, int PhaseStep);
	void GenerateSamplesVectorized(int SampleCount, double& CurChip, double CodeStep, double& CurPhase, double PhaseStep, double Amp);

};
