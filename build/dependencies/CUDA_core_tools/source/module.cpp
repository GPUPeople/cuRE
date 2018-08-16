


#include <CUDA/error.h>
#include <CUDA/module.h>


namespace CU
{
	unique_module loadModule(const void* image)
	{
		CUmodule module;
		succeed(cuModuleLoadData(&module, image));
		return unique_module(module);
	}

	unique_module loadModuleFile(const char* filename)
	{
		CUmodule module;
		succeed(cuModuleLoad(&module, filename));
		return unique_module(module);
	}

	CUfunction getFunction(CUmodule module, const char* name)
	{
		CUfunction function;
		succeed(cuModuleGetFunction(&function, module, name));
		return function;
	}

	CUfunction getFunction(CUmodule module, std::string_view name)
	{
		return getFunction(module, data(name));
	}

	CUdeviceptr getGlobal(CUmodule module, const char* name)
	{
		CUdeviceptr ptr;
		std::size_t size;
		succeed(cuModuleGetGlobal(&ptr, &size, module, name));
		return ptr;
	}

	CUdeviceptr getGlobal(CUmodule module, std::string_view name)
	{
		return getGlobal(module, data(name));
	}

	CUtexref getTextureReference(CUmodule module, const char* name)
	{
		CUtexref ref;
		succeed(cuModuleGetTexRef(&ref, module, name));
		return ref;
	}

	CUtexref getTextureReference(CUmodule module, std::string_view name)
	{
		return getTextureReference(module, data(name));
	}

	CUsurfref getSurfaceReference(CUmodule module, const char* name)
	{
		CUsurfref ref;
		succeed(cuModuleGetSurfRef(&ref, module, name));
		return ref;
	}

	CUsurfref getSurfaceReference(CUmodule module, std::string_view name)
	{
		return getSurfaceReference(module, data(name));
	}
}
