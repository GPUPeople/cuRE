


#ifndef INCLUDED_PLUGIN_PTR
#define INCLUDED_PLUGIN_PTR

#pragma once

#include <memory>

#include "PlugIn.h"


struct PlugInDeleter
{
	void operator ()(PlugIn* plugin)
	{
		plugin->destroy();
	}
};

template <class T>
using plugin_ptr = std::unique_ptr<T, PlugInDeleter>;

#endif  // INCLUDED_PLUGIN_PTR
