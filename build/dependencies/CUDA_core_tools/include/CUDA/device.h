


#ifndef INCLUDED_CUDA_DEVICE
#define INCLUDED_CUDA_DEVICE

#pragma once

#include <cstddef>
#include <string>

#include <cuda.h>

#include "error.h"


namespace CU
{
	CUdevice getDevice(int ordinal);
	
	CUdevice getDevice(CUcontext ctx);
	
	int getDeviceCount();
	
	std::string getDeviceName(CUdevice dev);
	
	template <CUdevice_attribute attribute>
	inline int getDeviceAttribute(CUdevice device)
	{
		int v;
		succeed(cuDeviceGetAttribute(&v, attribute, device));
		return v;
	}
	
	std::size_t getDeviceMemory(CUdevice dev);
	
	CUdevice findMatchingDevice(int cc_major, int cc_minor);
}

#endif  // INCLUDED_CUDA_DEVICE
