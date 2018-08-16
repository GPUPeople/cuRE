


#include <cassert>

#include <CUDA/error.h>
#include <CUDA/link.h>


namespace CU
{
	unique_link_state createLinker(unsigned int num_options, const CUjit_option* options, void* const * option_values)
	{
		CUlinkState state;
		succeed(cuLinkCreate(num_options, const_cast<CUjit_option*>(options), const_cast<void**>(option_values), &state));
		return unique_link_state(state);
	}
	
	unique_link_state createLinker(std::initializer_list<CUjit_option> options, std::initializer_list<void*> option_values)
	{
		assert(options.size() == option_values.size());
		return createLinker(static_cast<unsigned int>(options.size()), options.begin(), option_values.begin());
	}
	
	unique_link_state createLinker()
	{
		return createLinker(0U, nullptr, nullptr);
	}
}
