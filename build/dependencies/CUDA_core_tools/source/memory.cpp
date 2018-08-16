


#include <CUDA/error.h>
#include <CUDA/memory.h>


namespace CU
{
	unique_ptr allocMemory(std::size_t size)
	{
		CUdeviceptr ptr;
		succeed(cuMemAlloc(&ptr, size));
		return unique_ptr(ptr);
	}
	
	unique_ptr allocMemoryPitched(std::size_t& pitch, std::size_t row_size, std::size_t num_rows, unsigned int element_size)
	{
		CUdeviceptr ptr;
		succeed(cuMemAllocPitch(&ptr, &pitch, row_size, num_rows, element_size));
		return unique_ptr(ptr);
	}
	
	pitched_memory allocMemoryPitched(std::size_t row_size, std::size_t num_rows, unsigned int element_size)
	{
		CUdeviceptr ptr;
		std::size_t pitch;
		succeed(cuMemAllocPitch(&ptr, &pitch, row_size, num_rows, element_size));
		return pitched_memory(unique_ptr(ptr), pitch);
	}
}
