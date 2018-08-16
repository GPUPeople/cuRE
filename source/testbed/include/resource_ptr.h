


#ifndef INCLUDED_RESOURCE_PTR
#define INCLUDED_RESOURCE_PTR

#pragma once

#include <memory>

#include "Resource.h"


struct ResourceDeleter
{
	void operator ()(Resource* resource)
	{
		resource->destroy();
	}
};

template <class T>
using resource_ptr = std::unique_ptr<T, ResourceDeleter>;

#endif  // INCLUDED_RESOURCE_PTR
