


#ifndef INCLUDED_DISPLAY
#define INCLUDED_DISPLAY

#pragma once

#include <string>
#include <chrono>

#include <GL/buffer.h>
#include <GL/shader.h>
#include <GL/texture.h>
#include <GL/vertex_array.h>
#include <GL/framebuffer.h>

#include <GL/platform/Renderer.h>
#include <GL/platform/Context.h>
#include <GL/platform/Window.h>
#include <GL/platform/DefaultDisplayHandler.h>

#include <image.h>
#include <rgba8.h>

#include "Navigator.h"
#include "Camera.h"
#include "Scene.h"

#include "Renderer.h"


class Config;
class PerformanceMonitor;

class Display : private GL::platform::DefaultDisplayHandler
{
private:
	GL::platform::Window window;
	GL::platform::Context context;
	GL::platform::context_scope<GL::platform::Window> ctx;

	int buffer_width;
	int buffer_height;

	GL::VertexArray vao;

	GL::Texture color_buffer;

	GL::Sampler color_buffer_sampler;

	GL::VertexShader vs;
	GL::FragmentShader fs;
	GL::Program prog;

	Renderer* renderer;

	std::string renderer_name;

	std::chrono::steady_clock::time_point start;
	std::chrono::steady_clock::time_point next_fps_tick;

	PerformanceMonitor* perf_mon;

	unsigned int frame_count;

	void resize(int width, int height) override;

public:
	Display(const Display&) = delete;
	Display& operator =(const Display&) = delete;

	Display(const Config& config);

	void render(Scene& scene, Camera& camera);

	void resizeWindow(int width, int height);

	image2D<RGBA8> screenshot() const;

	void attach(Renderer* renderer, const char* name);
	void attach(PerformanceMonitor* performance_monitor);

	void attach(GL::platform::MouseInputHandler* mouse_input);
	void attach(GL::platform::KeyboardInputHandler* keyboard_input);

	void save(Config& config) const;
};

#endif  // INCLUDED_DISPLAY
