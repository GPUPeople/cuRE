


#ifndef INCLUDED_CONSOLE_HANDLER
#define INCLUDED_CONSOLE_HANDLER

#pragma once

#include <GL/platform/InputHandler.h>


class Navigator;
class RenderingSystem;
class PlugInManager;

class ConsoleHandler : public virtual GL::platform::ConsoleHandler
{
private:
	Navigator& navigator;
	RenderingSystem& rendering_system;
	PlugInManager& plugin_man;

public:
	ConsoleHandler(Navigator& navigator, RenderingSystem& rendering_system, PlugInManager& plugin_man);

	ConsoleHandler(const ConsoleHandler&) = delete;
	ConsoleHandler& operator =(const ConsoleHandler&) = delete;

	void command(const char* command, size_t length);
};

#endif  // INCLUDED_CONSOLE_HANDLER
