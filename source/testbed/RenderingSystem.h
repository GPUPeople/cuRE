


#ifndef INCLUDED_RENDERING_SYSTEM
#define INCLUDED_RENDERING_SYSTEM

#pragma once

#include <tuple>
#include <memory>
#include <vector>
#include <string>

#include <GL/platform/Renderer.h>

#include "Navigator.h"
#include "PerspectiveCamera.h"
#include "Noncamera.h"
#include "Scene.h"
#include "Display.h"


#include "interface.h"

#include "plugin_ptr.h"
#include "Renderer.h"

#include "PlugInManager.h"


class Config;
class PerformanceMonitor;

class RenderingSystem : public virtual GL::platform::Renderer, private virtual PlugInManager::Callback
{
private:
	Display display;

	PerspectiveCamera camera;
	//Noncamera camera;

	plugin_ptr<::Renderer> renderer;

	std::unique_ptr<Scene> scene;

	PlugInManager& plugin_man;

	PerformanceMonitor* perf_mon;

	typedef std::vector<std::tuple<std::string, createRendererFunc>> renderer_list_t;
	renderer_list_t renderers;
	size_t current_plugin;

	int device;

	const char* rendererName() const;
	void releaseRenderer();

	void onRendererPlugInLoaded(const char* name, createRendererFunc create_function);
	void onRendererPlugInUnloading(createRendererFunc create_function);
	void onDetach(PlugInManager* plugin_man) {};

public:
	RenderingSystem(const RenderingSystem&) = delete;
	RenderingSystem& operator =(const RenderingSystem&) = delete;

	RenderingSystem(PlugInManager& plugin_man, const Config& config, PerformanceMonitor* perf_mon, const char* scene = nullptr, int res_x = -1, int res_y = -1);
	~RenderingSystem();

	void switchRenderer(int d = 1);
	void switchRenderer(const char* name);

	void sceneButtonPushed(GL::platform::Key c);

	void render() override;

	void screenshot(const char* filename = nullptr) const;

	void attach(const Navigator* navigator);
	void attach(GL::platform::MouseInputHandler* mouse_input);
	void attach(GL::platform::KeyboardInputHandler* keyboard_input);

	void save(Config& config) const;
};

#endif  // INCLUDED_RENDERING_SYSTEM
