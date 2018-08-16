


#ifndef INCLUDED_CUDA_LINK
#define INCLUDED_CUDA_LINK

#pragma once

#include <cstddef>
#include <initializer_list>

#include <cuda.h>

#include <CUDA/unique_handle.h>

#include "error.h"


namespace CU
{
	struct LinkDestroyDeleter
	{
		void operator ()(CUlinkState state) const
		{
			cuLinkDestroy(state);
		}
	};
	
	using unique_link_state = unique_handle<CUlinkState, nullptr, LinkDestroyDeleter>;
	
	unique_link_state createLinker(unsigned int num_options, const CUjit_option* options, void* const * option_values);
	unique_link_state createLinker(std::initializer_list<CUjit_option> options, std::initializer_list<void*> option_values);
	unique_link_state createLinker();
	
	template <std::size_t N>
	inline unique_link_state createLinker(CUjit_option (&options)[N], void* (&option_values)[N])
	{
		return createLinker(N, options, option_values);
	}
}

#endif  // INCLUDED_CUDA_LINK
