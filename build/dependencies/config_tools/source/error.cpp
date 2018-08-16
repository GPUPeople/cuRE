


#include <config/error.h>


namespace config
{
	const char* not_found::what() const noexcept
	{
		return "value/node not found";
	}
}
