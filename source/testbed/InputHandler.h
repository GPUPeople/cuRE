


#ifndef INCLUDED_INPUT_HANDLER
#define INCLUDED_INPUT_HANDLER

#pragma once

#include <GL/platform/InputHandler.h>


class Navigator;
class RenderingSystem;
class PlugInManager;

class InputHandler : public virtual GL::platform::KeyboardInputHandler, public virtual GL::platform::MouseInputHandler
{
	Navigator& navigator;
	RenderingSystem& rendering_system;
	PlugInManager& plugin_man;

	bool shift = false;

public:
	InputHandler(Navigator& navigator, RenderingSystem& display, PlugInManager& plugin_man);

	InputHandler(const InputHandler&) = delete;
	InputHandler& operator =(const InputHandler&) = delete;

	void keyDown(GL::platform::Key key);
	void keyUp(GL::platform::Key key);
	void buttonDown(GL::platform::Button button, int x, int y);
	void buttonUp(GL::platform::Button button, int x, int y);
	void mouseMove(int x, int y);
	void mouseWheel(int delta);
};

#endif  // INCLUDED_INPUT_HANDLER
