


#include <iostream>

#include <CUDA/error.h>
#include <CUDA/device.h>
#include <CUDA/context.h>
#include <CUDA/module.h>

#include "PipelineModule.h"
#include "Pipeline.h"


namespace cuRE
{
	PipelineKernel::PipelineKernel(CUmodule module, const char* symbol_name, int rasterizer_count, int warps_per_block, int virtual_rasterizers)
		: kernel(CU::getFunction(module, symbol_name)),
		  preparenewprimitive(CU::getFunction(module, "prepareRasterizationNewPrimitive")),
		  geometry_producing_blocks_count_symbol(CU::getGlobal(module, "geometryProducingBlocksCount")),
		  rasterizer_count(rasterizer_count),
		  warps_per_block(warps_per_block),
		  virtual_rasterizers(virtual_rasterizers),
		  symbol_name(symbol_name)
	{
	}

	std::complex<int> PipelineKernel::computeOccupancy() const
	{
		int num_multiprocessors;
		succeed(cuDeviceGetAttribute(&num_multiprocessors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, CU::getDevice(CU::getCurrentContext())));

		int blocks_per_multiprocessor;
		succeed(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_multiprocessor, kernel, warps_per_block * 32U, 0U));

		int num_blocks = num_multiprocessors * blocks_per_multiprocessor;

		if (num_blocks < rasterizer_count)
			return std::complex<int>(0, rasterizer_count);

		return std::complex<int>(num_blocks, rasterizer_count);
	}

	void PipelineKernel::prepare() const
	{
		//succeed(cuMemcpyHtoD(geometry_producing_blocks_count_symbol, &rasterizer_count, 4));
		//succeed(cuLaunchKernel(preparenewprimitive, 1U, 1U, 1U, 1024U, 1U, 1U, 0U, 0, nullptr, nullptr));
		preparenewprimitive({ 1U, 1U, 1U }, { 1024U, 1U, 1U }, 0U, nullptr, rasterizer_count);
	}

	int PipelineKernel::launch() const
	{
		kernel({ static_cast<unsigned int>(rasterizer_count), 1U, 1U }, { warps_per_block * 32U, 1U, 1U }, 0U, nullptr);
		return rasterizer_count;
	}

	std::ostream& PipelineKernel::printInfo(std::ostream& out) const
	{
		int num_regs;
		succeed(cuFuncGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, kernel));

		int shared_mem;
		succeed(cuFuncGetAttribute(&shared_mem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel));

		int const_mem;
		succeed(cuFuncGetAttribute(&const_mem, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, kernel));

		int local_mem;
		succeed(cuFuncGetAttribute(&local_mem, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel));

		std::cout << symbol_name << ":\n"
		          << "  regs: " << num_regs << " smem: " << shared_mem << " cmem: " << const_mem << " lmem: " << local_mem << " -> " << rasterizer_count << " blocks\n";

		auto occupancy = computeOccupancy();

		if (occupancy.real() > occupancy.imag())
			std::cout << "WARNING: suboptimal launch configuration; could fit more rasterizers (" << occupancy.real() << ") on GPU\n";
		else if (occupancy.real() < occupancy.imag())
			std::cout << "ERROR: pipeline kernel cannot be launched; too many rasterizers for this GPU\n";

		return out;
	}
}
