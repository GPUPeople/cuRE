


#ifndef INCLUDED_RESOURCE_IMP
#define INCLUDED_RESOURCE_IMP

#pragma once

#include "Renderer.h"


template <class T>
class ResourceImp : public virtual T
{
	template <typename... Args>
	ResourceImp(Args&&... args)
		: T(args...)
	{
	}

	~ResourceImp() = default;

public:
	ResourceImp(const ResourceImp&) = delete;
	ResourceImp& operator =(const ResourceImp&) = delete;

	template <typename... Args>
	static T* create(Args&&... args)
	{
		return new ResourceImp(args...);
	}

	void destroy()
	{
		delete this;
	}
};


#endif  // INCLUDED_RESOURCE_IMP
