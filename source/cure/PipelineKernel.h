


#ifndef INCLUDED_CURE_PIPELINE_KERNEL
#define INCLUDED_CURE_PIPELINE_KERNEL

#pragma once

#include <cstdint>
#include <iosfwd>
#include <complex>

#include <cuda.h>
#include <CUDA/launch.h>


namespace cuRE
{
	class PipelineKernel
	{
		CU::Function<> kernel;
		CU::Function<std::uint32_t> preparenewprimitive;
		CUdeviceptr geometry_producing_blocks_count_symbol;

		int rasterizer_count;
		int warps_per_block;
		int virtual_rasterizers;
		std::string symbol_name;

	public:
		PipelineKernel(CUmodule module, const char* symbol_name, int rasterizer_count, int warps_per_block, int virtual_rasterizers);

		std::complex<int> computeOccupancy() const;

		void prepare() const;

		int launch() const;

		std::ostream& printInfo(std::ostream& out) const;
	};
}

#endif  // INCLUDED_CURE_PIPELINE_KERNEL
