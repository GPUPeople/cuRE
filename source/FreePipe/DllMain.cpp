


#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <PlugInClient.h>

#include "Renderer.h"


BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID)
{
	return TRUE;
}

void __stdcall registerPlugInServer(PlugInClient* client)
{
	client->registerRenderer("FreePipe", &FreePipe::Renderer::create);
}
