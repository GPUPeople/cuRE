


#include <cassert>
#include <stdexcept>

#include "Navigator.h"
#include "RenderingSystem.h"
#include "PlugInManager.h"

#include "InputHandler.h"


namespace
{
	Navigator::Button translateButton(GL::platform::Button button)
	{
		switch (button)
		{
		case GL::platform::Button::LEFT:
			return Navigator::Button::LEFT;
		case GL::platform::Button::RIGHT:
			return Navigator::Button::RIGHT;
		case GL::platform::Button::MIDDLE:
			return Navigator::Button::MIDDLE;
		}
		assert(false);
		return static_cast<Navigator::Button>(-1);
	}
}

InputHandler::InputHandler(Navigator& navigator, RenderingSystem& rendering_system, PlugInManager& plugin_man)
	: navigator(navigator),
	  rendering_system(rendering_system),
	  plugin_man(plugin_man)
{
}

void InputHandler::keyDown(GL::platform::Key key)
{
	if (key == GL::platform::Key::SHIFT)
		shift = true;
}

void InputHandler::keyUp(GL::platform::Key key)
{
	switch (key)
	{
	case GL::platform::Key::SHIFT:
		shift = false;
		break;

	case GL::platform::Key::F5:
		plugin_man.refreshModules();
		break;

	case GL::platform::Key::F8:
		rendering_system.screenshot();
		break;

	case GL::platform::Key::TAB:
		rendering_system.switchRenderer(shift ? -1 : 1);
		break;

	case GL::platform::Key::BACKSPACE:
		navigator.reset();
		break;

	default:
		rendering_system.sceneButtonPushed(key);
	}
}

void InputHandler::buttonDown(GL::platform::Button button, int x, int y)
{
	navigator.buttonDown(translateButton(button), x, y);
}

void InputHandler::buttonUp(GL::platform::Button button, int x, int y)
{
	navigator.buttonUp(translateButton(button), x, y);
}

void InputHandler::mouseMove(int x, int y)
{
	navigator.mouseMove(x, y);
}

void InputHandler::mouseWheel(int delta)
{
	navigator.mouseWheel(delta);
}
