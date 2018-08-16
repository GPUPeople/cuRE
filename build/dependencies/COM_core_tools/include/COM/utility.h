


#ifndef INCLUDED_COM_UTILITY
#define INCLUDED_COM_UTILITY

#pragma once

#include <win32/platform.h>
#include <Unknwn.h>

#include "error.h"
#include "unique_ptr.h"


namespace COM
{
	template <typename T>
	constexpr const GUID& getIID() = delete;

	template <typename T>
	constexpr const GUID& iidof = getIID<T>();

	template <typename T, const IID& iid = iidof<T>>
	inline unique_ptr<T> getInterface(IUnknown* obj)
	{
		void* p;
		throw_error(obj->QueryInterface(iid, &p));
		return make_unique_ptr(static_cast<T*>(p));
	}
}

#endif  // INCLUDED_COM_UTILITY
