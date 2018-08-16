


#ifndef INCLUDED_PLUGIN
#define INCLUDED_PLUGIN

#pragma once

#include <interface.h>


struct INTERFACE PlugIn
{
	virtual void destroy() = 0;

protected:
	PlugIn() = default;
	PlugIn(const PlugIn&) = default;
	PlugIn& operator =(const PlugIn&) = default;
	~PlugIn() = default;
};

#endif  // INCLUDED_PLUGIN
