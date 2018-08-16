


#ifndef INCLUDED_CUDA_MODULE
#define INCLUDED_CUDA_MODULE

#pragma once

#include <string_view>

#include <cuda.h>

#include <CUDA/unique_handle.h>


namespace CU
{
	struct ModuleUnloadDeleter
	{
		void operator ()(CUmodule module) const
		{
			cuModuleUnload(module);
		}
	};
	
	using unique_module = unique_handle<CUmodule, nullptr, ModuleUnloadDeleter>;
	
	unique_module loadModuleFile(const char* filename);
	unique_module loadModule(const void* image);
	
	CUfunction getFunction(CUmodule module, const char* name);
	CUfunction getFunction(CUmodule module, std::string_view name);
	CUdeviceptr getGlobal(CUmodule module, const char* name);
	CUdeviceptr getGlobal(CUmodule module, std::string_view name);
	CUtexref getTextureReference(CUmodule module, const char* name);
	CUtexref getTextureReference(CUmodule module, std::string_view name);
	CUsurfref getSurfaceReference(CUmodule module, const char* name);
	CUsurfref getSurfaceReference(CUmodule module, std::string_view name);


	template <CUfunction_attribute attribute>
	inline int getFunctionAttribute(CUfunction function)
	{
		int v;
		succeed(cuFuncGetAttribute(&v, attribute, function));
		return v;
	}
}

#endif  // INCLUDED_CUDA_MODULE
