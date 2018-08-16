#pragma once

#include <iostream>
#include <cuda.h>
#include <CUDA/module.h>
#include <CUDA/error.h>
#include "Math.hpp"
#include "Array.hpp"

#define GL_SHADER_SOURCE(CODE) #CODE 

#define fail(A) printf("Error at line %d in %s: %s", __LINE__, __FILE__, A); std::cin.ignore();

struct Param // Wrapper for converting kernel parameters to CUDA-compatible types.
{
	FW::S32         size;
	FW::S32         align;
	const void*     value;
	template <class T>  Param(const T& v)                { size = sizeof(T); align = __alignof(T); value = &v; }
};

typedef const Param& P; // To reduce the amount of code in setParams() overloads.

void  setParams(const Param* const* params, int numParams, FW::Array<FW::U8>& m_params);

class Buffer
{
public:

	Buffer() : address(0x0), size(0)
	{}
	
	CUdeviceptr address;
	
	size_t size;
};

void resizeDiscard(Buffer& buffer, size_t size);

void launchKernel(CUfunction function, size_t numThreads, FW::Vec2i blockSize);

void launchKernelGrid(CUfunction function, FW::Vec2i gridSize, FW::Vec2i blockSize);

void launchParamlessKernelPreferShared(CUfunction function, FW::Vec2i numThreads, FW::Vec2i blockSize);

void launchParamlessKernelPreferShared(CUfunction function, size_t numThreads, FW::Vec2i blockSize);

void launchParamlessKernelPreferSharedGrid(CUfunction function, FW::Vec2i gridSize, FW::Vec2i blockSize);

CUdeviceptr getGlobalPtr(CUmodule module, const char* varname);

void setTexRef(const char* name, CUmodule module, CUdeviceptr ptr, FW::S64 size, CUarray_format format, int numComponents);

void setSurfRef(const char* name, CUmodule module, CUarray cudaArray);

template<typename T>
void getGlobal(CUmodule module, const char* varname, T& var)
{
	CUdeviceptr ptr = getGlobalPtr(module, varname);
	succeed(cuMemcpyDtoH(&var, ptr, sizeof(T)));
}

template<typename T>
class MutableMem
{
private:
	T* host_mem;

	size_t count;

	CUdeviceptr device_address;

	MutableMem(const MutableMem<T>&);

public:

	T* get()
	{
		return host_mem;
	}

	MutableMem(CUdeviceptr ptr, size_t num)
	{
		count = num;
		host_mem = new T[num];
		device_address = ptr;
		succeed(cuMemcpyDtoH(host_mem, ptr, sizeof(T)*num));
	}

	~MutableMem()
	{
		succeed(cuMemcpyHtoD(device_address, host_mem, count*sizeof(T)));

		if(host_mem)
		{	delete[] host_mem;	}
	}
};

template<typename T>
class MutableVar
{
private:
	T host_var;

	CUdeviceptr device_address;

	MutableVar(const MutableVar<T>&);

public:

	T& get()
	{	return host_var;	}

	MutableVar(CUmodule module, const char* name)
	{
		size_t size;
		succeed(cuModuleGetGlobal(&device_address, &size, module, name));

		if (size != sizeof(T))
		{	fail("Wrong variable type was found!");	}

		succeed(cuMemcpyDtoH(&host_var, device_address, sizeof(T)));
	}

	~MutableVar()
	{	succeed(cuMemcpyHtoD(device_address, &host_var, sizeof(T)));	}
};

