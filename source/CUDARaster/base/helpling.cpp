#include "helpling.h"
#include "Array.hpp"
#include <CUDA/error.h>

void resizeDiscard(Buffer& buffer, size_t size)
{
	if(size == buffer.size)
	{	return;	}

	if(buffer.address)
	{	succeed(cuMemFree(buffer.address));	}

	succeed(cuMemAlloc(&buffer.address, size));

	buffer.size = size;
}

CUdeviceptr getGlobalPtr(CUmodule module, const char* varname)
{
	CUdeviceptr ptr;
	size_t size;
	succeed(cuModuleGetGlobal(&ptr, &size, module, varname));
	return ptr;
}

void launchKernelGrid(CUfunction function, FW::Vec2i gridSize, FW::Vec2i blockSize)
{
	//succeed(cuFuncSetCacheConfig(function, CU_FUNC_CACHE_PREFER_SHARED));
	//succeed(cuFuncSetSharedMemConfig(function, CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE));
	succeed(cuFuncSetBlockShape(function, blockSize.x, blockSize.y, 1));
	succeed(cuLaunchGrid(function, gridSize.x, gridSize.y));
}

void launchKernel(CUfunction function, size_t numThreads, FW::Vec2i blockSize)
{
	blockSize = (min(blockSize) > 0) ? blockSize : FW::Vec2i(32, 4);
	int maxGridWidth = 65536;

	CUdevice dev;
	cuDeviceGet(&dev, 0);

	int tmp;
	succeed(cuDeviceGetAttribute(&tmp, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev));
	if (tmp != 0)
	{
		maxGridWidth = tmp;
	}

	int threadsPerBlock = blockSize.x * blockSize.y;

	FW::Vec2i gridSize = FW::Vec2i((numThreads + threadsPerBlock - 1) / threadsPerBlock, 1);
	while (gridSize.x > maxGridWidth)
	{
		gridSize.x = (gridSize.x + 1) >> 1;
		gridSize.y <<= 1;
	}

	launchKernelGrid(function, gridSize, blockSize);
}

void launchParamlessKernelPreferSharedGrid(CUfunction function, FW::Vec2i gridSize, FW::Vec2i blockSize)
{
	succeed(cuParamSetSize(function, 0));
	succeed(cuFuncSetCacheConfig(function, CU_FUNC_CACHE_PREFER_SHARED));
	succeed(cuFuncSetSharedMemConfig(function, CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE));
	succeed(cuFuncSetBlockShape(function, blockSize.x, blockSize.y, 1));
	succeed(cuLaunchGrid(function, gridSize.x, gridSize.y));
}

void launchParamlessKernelPreferShared(CUfunction function, size_t numThreads, FW::Vec2i blockSize)
{
	blockSize = (min(blockSize) > 0) ? blockSize : FW::Vec2i(32, 4);
	int maxGridWidth = 65536;

	CUdevice dev;
	cuDeviceGet(&dev, 0);

	int tmp;
	succeed(cuDeviceGetAttribute(&tmp, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev));
	if (tmp != 0)
	{	maxGridWidth = tmp;	}

	int threadsPerBlock = blockSize.x * blockSize.y;
	
	FW::Vec2i gridSize = FW::Vec2i((numThreads + threadsPerBlock - 1) / threadsPerBlock, 1);
	while (gridSize.x > maxGridWidth)
	{
		gridSize.x = (gridSize.x + 1) >> 1;
		gridSize.y <<= 1;
	}

	launchParamlessKernelPreferSharedGrid(function, gridSize, blockSize);
}

void launchParamlessKernelPreferShared(CUfunction function, FW::Vec2i numThreads, FW::Vec2i blockSize)
{
	FW::Vec2i gridSize;
	blockSize = (min(blockSize) > 0) ? blockSize : FW::Vec2i(32, 4);
	gridSize = (numThreads + blockSize - 1) / blockSize;

	launchParamlessKernelPreferSharedGrid(function, gridSize, blockSize);
}

void setSurfRef(const char* name, CUmodule module, CUarray cudaArray)
{
	CUsurfref surfRef;
	succeed(cuModuleGetSurfRef(&surfRef, module, name));
	succeed(cuSurfRefSetArray(surfRef, cudaArray, 0));
}

void setTexRef(const char* name, CUmodule module, CUdeviceptr ptr, FW::S64 size, CUarray_format format, int numComponents)
{
	CUtexref texRef;
	succeed(cuModuleGetTexRef(&texRef, module, name));
	succeed(cuTexRefSetFormat(texRef, format, numComponents));
	succeed(cuTexRefSetAddress(NULL, texRef, ptr, (FW::U32)size));
}


void setParams(const Param* const* params, int numParams, FW::Array<FW::U8>& m_params)
{
	int size = 0;
	for (int i = 0; i < numParams; i++)
	{
		size = (size + params[i]->align - 1) & -params[i]->align;
		size += params[i]->size;
	}
	m_params.reset(size);

	int ofs = 0;
	for (int i = 0; i < numParams; i++)
	{
		ofs = (ofs + params[i]->align - 1) & -params[i]->align;
		memcpy(m_params.getPtr(ofs), params[i]->value, params[i]->size);
		ofs += params[i]->size;
	}
}