


#include <sstream>
#include <iomanip>
#include <string>

#include <GL/error.h>

#include "Config.h"
#include "Display.h"

#include "PerformanceMonitor.h"


constexpr bool FRAMEBUFFER_SRGB = true;


extern const char fullscreen_triangle_vs[];
extern const char draw_color_buffer_fs[];

namespace
{
	class context_scope
	{
	private:
		RendereringContext* context;

	public:
		context_scope(Renderer* renderer)
			: context(renderer->beginFrame())
		{
		}

		~context_scope()
		{
			context->finish();
		}

		const RendereringContext* operator ->() const { return context; }
		RendereringContext* operator ->() { return context; }

		operator const RendereringContext* () const { return context; }
		operator RendereringContext* () { return context; }
	};


	void APIENTRY OpenGLDebugOutputCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam)
	{
		OutputDebugStringA(message);
		OutputDebugStringA("\n");
	}

	WINDOWPLACEMENT loadWindowPlacement(const Config& config)
	{
		WINDOWPLACEMENT window_placement;
		window_placement.length = sizeof(WINDOWPLACEMENT);
		window_placement.flags = config.loadInt("flags", 0U);
		window_placement.showCmd = config.loadInt("show", SW_SHOWNORMAL);
		window_placement.ptMinPosition.x = config.loadInt("min_x", 0);
		window_placement.ptMinPosition.y = config.loadInt("min_y", 0);
		window_placement.ptMaxPosition.x = config.loadInt("max_x", 0);
		window_placement.ptMaxPosition.y = config.loadInt("max_y", 0);
		window_placement.rcNormalPosition.left = config.loadInt("left", 0);
		window_placement.rcNormalPosition.top = config.loadInt("top", 0);
		window_placement.rcNormalPosition.right = config.loadInt("right", 800);
		window_placement.rcNormalPosition.bottom = config.loadInt("bottom", 600);
		return window_placement;
	}

	void saveWindowPlacement(Config& config, const WINDOWPLACEMENT& window_placement)
	{
		config.saveInt("flags", window_placement.flags);
		config.saveInt("show", window_placement.showCmd);
		config.saveInt("min_x", window_placement.ptMinPosition.x);
		config.saveInt("min_y", window_placement.ptMinPosition.y);
		config.saveInt("max_x", window_placement.ptMaxPosition.x);
		config.saveInt("max_y", window_placement.ptMaxPosition.y);
		config.saveInt("left", window_placement.rcNormalPosition.left);
		config.saveInt("top", window_placement.rcNormalPosition.top);
		config.saveInt("right", window_placement.rcNormalPosition.right);
		config.saveInt("bottom", window_placement.rcNormalPosition.bottom);
	}
}

Display::Display(const Config& config)
	: window("No Renderer", loadWindowPlacement(config.loadConfig("window_placement")), 0, 0, false),
	  //context(window.createContext(4, 5, false)),
	  context(window.createContext(4, 5, true)),
	  ctx(context, window),
	  renderer(nullptr),
	  vs(GL::compileVertexShader(fullscreen_triangle_vs)),
	  fs(GL::compileFragmentShader(draw_color_buffer_fs)),
	  start(std::chrono::steady_clock::now()),
	  next_fps_tick(start),
	  frame_count(0U),
	  perf_mon(nullptr)
{
	glAttachShader(prog, vs);
	glAttachShader(prog, fs);
	GL::linkProgram(prog);

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CW);

	if constexpr (FRAMEBUFFER_SRGB)
		glEnable(GL_FRAMEBUFFER_SRGB);

	GL::throw_error();

	window.attach(this);

	glDebugMessageCallback(OpenGLDebugOutputCallback, nullptr);
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);

	ctx.setSwapInterval(0);
}

void Display::attach(Renderer* renderer, const char* name)
{
	if (renderer)
	{
		renderer_name = name;
		renderer->setRenderTarget(color_buffer, buffer_width, buffer_height);
	}
	else
		renderer_name = "No Renderer";

	window.title(renderer_name.c_str());

	next_fps_tick = std::chrono::steady_clock::now() + std::chrono::seconds(1);
	frame_count = 0U;

	this->renderer = renderer;
}

void Display::resize(int width, int height)
{
	buffer_width = width;
	buffer_height = height;

	auto new_color_buffer = GL::createTexture2D(buffer_width, buffer_height, 1, FRAMEBUFFER_SRGB ? GL_SRGB8_ALPHA8 : FRAMEBUFFER_SRGB);

	if (renderer)
		renderer->setRenderTarget(new_color_buffer, buffer_width, buffer_height);

	color_buffer = std::move(new_color_buffer);

	GL::throw_error();
}

void Display::attach(PerformanceMonitor* performance_monitor)
{
	perf_mon = performance_monitor;
}

void Display::render(Scene& scene, Camera& camera)
{
	auto now = std::chrono::steady_clock::now();

	if (renderer)
	{
		context_scope context(renderer);
		context->clearColorBuffer(0.6f, 0.7f, 1.0f, 1.0f);
		//context->clearColorBuffer(0.0f, 0.0f, 0.0f, 1.0f);
		//context->clearColorBuffer(1.0f, 1.0f, 1.0f, 1.0f);
		context->clearDepthBuffer(1.0f);
		context->setViewport(0.0f, 0.0f, static_cast<float>(buffer_width), static_cast<float>(buffer_height));
		//context->setViewport(0.5f*static_cast<float>(buffer_width), 0.5f*static_cast<float>(buffer_height), 0.5f*static_cast<float>(buffer_width), 0.5f*static_cast<float>(buffer_height));
		Camera::UniformBuffer camera_uniform_buffer;
		camera.writeUniformBuffer(&camera_uniform_buffer, buffer_width * 1.0f / buffer_height);
		context->setCamera(camera_uniform_buffer);
		context->setUniformf(0, std::chrono::duration<float>(now - start).count());
		scene.update(camera_uniform_buffer);
		scene.draw(context);
	}

	glDisable(GL_DEPTH_TEST);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	glViewport(0, 0, buffer_width, buffer_height);

	glActiveTexture(GL_TEXTURE0);
	glBindSampler(0, color_buffer_sampler);
	glBindTexture(GL_TEXTURE_2D, color_buffer);
	glBindVertexArray(vao);
	glUseProgram(prog);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	GL::throw_error();

	ctx.swapBuffers();

	++frame_count;

	if (now >= next_fps_tick)
	{
		//auto dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - next_fps_tick).count() + 1.0f;
		//float fps = frame_count * 1.0f / dt;

		std::ostringstream title;
		//title << std::fixed << std::setprecision(1) << renderer_name << " @ " << fps << " fps";
		title << renderer_name;
		if (perf_mon)
		{
			title << "    ";
			perf_mon->printStatus(title);
			perf_mon->reset();
		}
		window.title(title.str().c_str());

		next_fps_tick = now + std::chrono::seconds(1);
		frame_count = 0U;
	}
}

void Display::resizeWindow(int width, int height)
{
	window.resize(width, height);
}

image2D<RGBA8> Display::screenshot() const
{
	image2D<RGBA8> buffer(buffer_width, buffer_height);

	//glBindTexture(GL_TEXTURE_2D, color_buffer);
	//glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, data(buffer));
	glReadBuffer(GL_FRONT);
	glReadPixels(0, 0, buffer_width, buffer_height, GL_RGBA, GL_UNSIGNED_BYTE, data(buffer));
	GL::throw_error();

	using std::swap;
	for (int y = 0; y < height(buffer) / 2; ++y)
		for (int x = 0; x < width(buffer); ++x)
			std::swap(buffer(x, y), buffer(x, height(buffer) - y - 1));

	return buffer;
}

void Display::attach(GL::platform::MouseInputHandler* mouse_input)
{
	window.attach(mouse_input);
}

void Display::attach(GL::platform::KeyboardInputHandler* keyboard_input)
{
	window.attach(keyboard_input);
}

void Display::save(Config& config) const
{
	WINDOWPLACEMENT placement;
	window.savePlacement(placement);
	saveWindowPlacement(config.loadConfig("window_placement"), placement);
}
