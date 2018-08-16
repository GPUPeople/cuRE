


#ifndef INCLUDED_CURE_PIPELINE_MODULE
#define INCLUDED_CURE_PIPELINE_MODULE

#pragma once

#include <tuple>
#include <vector>

#include <CUDA/module.h>

#include "PipelineKernel.h"


namespace cuRE
{
	class PipelineModule
	{
	private:
		CU::unique_module module;

		std::vector<std::tuple<std::string, PipelineKernel>> pipeline_kernels;

		decltype(pipeline_kernels)::const_iterator findKernel(const char* name) const;
		decltype(pipeline_kernels)::iterator findKernel(const char* name);

	public:
		PipelineModule(const PipelineModule&) = delete;
		PipelineModule& operator =(const PipelineModule&) = delete;

		PipelineModule();

		CUdeviceptr getGlobal(const char* name) const;
		CUfunction getFunction(const char* name) const;
		CUtexref getTextureReference(const char* name) const;
		CUsurfref getSurfaceReference(const char* name) const;

		PipelineKernel findPipelineKernel(const char* name) const;
	};
}

#endif  // INCLUDED_CURE_PIPELINE_MODULE
