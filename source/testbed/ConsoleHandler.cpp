


#include "Navigator.h"
#include "RenderingSystem.h"
#include "PlugInManager.h"

#include "ConsoleHandler.h"


ConsoleHandler::ConsoleHandler(Navigator& navigator, RenderingSystem& rendering_system, PlugInManager& plugin_man)
	: navigator(navigator),
	  rendering_system(rendering_system),
	  plugin_man(plugin_man)
{
}

void ConsoleHandler::command(const char* command, size_t length)
{

}
