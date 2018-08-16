


#ifndef INCLUDED_PLUGIN_CLIENT
#define INCLUDED_PLUGIN_CLIENT

#pragma once

#include <interface.h>

#include "Renderer.h"


struct INTERFACE PlugInClient
{
	virtual void registerRenderer(const char* name, createRendererFunc create_function) = 0;

protected:
	PlugInClient() = default;
	PlugInClient(const PlugInClient&) = default;
	PlugInClient& operator =(const PlugInClient&) = default;
	~PlugInClient() = default;
};

using registerPlugInServerFunc = void (__stdcall*)(PlugInClient* client);

#endif  // INCLUDED_PLUGIN_CLIENT
