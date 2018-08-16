


#ifndef INCLUDED_CUDA_MEMORY
#define INCLUDED_CUDA_MEMORY

#pragma once

#include <cstddef>

#include <cuda.h>

#include <CUDA/unique_handle.h>


namespace CU
{
	struct MemFreeDeleter
	{
		void operator ()(CUdeviceptr ptr) const
		{
			cuMemFree(ptr);
		}
	};
	
	using unique_ptr = unique_handle<CUdeviceptr, 0ULL, MemFreeDeleter>;
	
	
	struct pitched_memory
	{
		pitched_memory(const pitched_memory&) = delete;
		pitched_memory& operator =(const pitched_memory&) = delete;
		
		unique_ptr memory;
		std::size_t pitch;
		
		pitched_memory() {}
		
		pitched_memory(unique_ptr memory, std::size_t pitch)
			: memory(std::move(memory)),
			  pitch(pitch)
		{
		}
		
		pitched_memory(pitched_memory&& m)
			: memory(std::move(m.memory)),
			  pitch(m.pitch)
		{
		}
		
		pitched_memory& operator =(pitched_memory&& m)
		{
			using std::swap;
			swap(memory, m.memory);
			pitch = m.pitch;
			return *this;
		}
	};
	
	
	unique_ptr allocMemory(std::size_t size);
	unique_ptr allocMemoryPitched(std::size_t& pitch, std::size_t row_size, std::size_t num_rows, unsigned int element_size);
	pitched_memory allocMemoryPitched(std::size_t row_size, std::size_t num_rows, unsigned int element_size);
}

#endif  // INCLUDED_CUDA_MEMORY
