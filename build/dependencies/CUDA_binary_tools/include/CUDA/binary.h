


#ifndef INCLUDED_CUDA_BINARY
#define INCLUDED_CUDA_BINARY

#pragma once

#include <vector>
#include <tuple>


namespace CU
{
	std::tuple<int, int> readComputeCapability(const char* image);
	std::vector<const char*> readSymbols(const char* image);
}

#endif  // INCLUDED_CUDA_BINARY
