

// CSatIfSignalCUDA.h
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#include <memory>
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <stdexcept>

#include "BasicTypes.h"
#include "PrnGenerate.h"
#include "NavBit.h"
#include "SatelliteSignal.h"

class CSatIfSignalCUDA {
public:
    CSatIfSignalCUDA(int sampleNumber, int ifFreq, GnssSystem system, int signalIndex, unsigned char svid);
    ~CSatIfSignalCUDA();
    void InitState(GNSS_TIME CurTime, PSATELLITE_PARAM pSatParam, NavBit* pNavData);
	void GetIfSample(GNSS_TIME CurTime);
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

	cuComplex GetPrnValue(double &CurChip, double CodeStep);
	cuComplex GetRotateValue(double & CurPhase, double PhaseStep);
	cuComplex GetRotateValue(unsigned int & CurPhase, int PhaseStep);
	void GenerateSamplesVectorized(int SampleCount, double& CurChip, double CodeStep, double& CurPhase, double PhaseStep, double Amp);

};
