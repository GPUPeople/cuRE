


#ifndef INCLUDED_CONFIG_ERROR
#define INCLUDED_CONFIG_ERROR

#pragma once

#include <exception>


namespace config
{
	struct not_found : std::exception
	{
		const char* what() const noexcept override;
	};
}

#endif  // INCLUDED_CONFIG_ERROR
