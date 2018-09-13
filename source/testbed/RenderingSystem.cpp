


#include <cstring>
#include <string>
#include <string_view>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <CUDA/device.h>
#include <CUDA/error.h>

#include <png.h>

#include "Config.h"
#include "RenderingSystem.h"

#include "BlendIsoScene.h"
#include "BlendScene.h"
#include "CheckerboardScene.h"
#include "ClipspaceScene.h"
#include "ClipspaceViewScene.h"
#include "CubeScene.h"
#include "EyeCandyScene.h"
#include "GlyphScene.h"
#include "IcosahedronScene.h"
#include "LoadedScene.h"
#include "PerformanceMonitor.h"
#include "SphereScene.h"
#include "StippleIsoScene.h"
#include "TriangleScene.h"
#include "WaterScene.h"

extern "C" __declspec(dllexport) DWORD NvOptimusEnablement = 0x1U;


namespace
{
	bool checkExtension(const char* filename, const char* extension)
	{
		const char* begin = std::strrchr(filename, extension[0]);

		if (begin)
			return std::strcmp(begin, extension) == 0;

		return false;
	}

	std::string buildModuleName(std::string_view name, int cc_major, int cc_minor)
	{
		std::ostringstream s;
		s << name << "_sm" << cc_major << cc_minor;
		return s.str();
	}

	void loadModules(PlugInManager& plugin_man, const Config& config, CUdevice device)
	{
		auto modules = config.loadTuple("modules", {});

		if (!modules.empty())
		{
			for (auto&& m : modules)
				plugin_man.loadModule(m.c_str());
		}
		else
		{
			int cc_major = CU::getDeviceAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>(device);
			int cc_minor = CU::getDeviceAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>(device);

			plugin_man.loadModule("GLRenderer");
			plugin_man.loadModule(buildModuleName("cure", cc_major, cc_minor).c_str());
			plugin_man.loadModule(buildModuleName("FreePipe", cc_major, cc_minor).c_str());

			if (cc_major <= 3 && cc_minor <= 5)
				plugin_man.loadModule(buildModuleName("CUDARaster", cc_major, cc_minor).c_str());
		}
	}
}

RenderingSystem::RenderingSystem(PlugInManager& plugin_man, const Config& config, PerformanceMonitor* perf_mon, const char* scene, int res_x, int res_y)
	: camera(60.0f * math::constants<float>::pi() / 180.0f, 0.1f, 100.0f),
	  renderer(nullptr),
	  display(config),
	  plugin_man(plugin_man),
	  current_plugin(0U),
	  device(config.loadInt("device", 0)),
	  perf_mon(perf_mon)
{
	plugin_man.attach(this);

	loadModules(plugin_man, config, device);

	display.attach(perf_mon);

	if (scene == nullptr || std::strncmp(scene, "icosahedron", 9) == 0)
		this->scene = std::make_unique<IcosahedronScene>();
	else if (std::strncmp(scene, "sphere", 6) == 0)
		this->scene = std::make_unique<SphereScene>();
	else if (std::strncmp(scene, "checkerboard_rendering_demo", 13) == 0)
		this->scene = std::make_unique<CheckerboardScene>(config, display);
	else if (std::strncmp(scene, "water", 5) == 0)
		this->scene = std::make_unique<WaterScene>(config);
	else if (std::strncmp(scene, "blend", 5) == 0)
		this->scene = std::make_unique<BlendScene>();
	else if (std::strncmp(scene, "isoblend", 8) == 0)
		this->scene = std::make_unique<BlendIsoScene>();
	else if (std::strncmp(scene, "isostipple", 10) == 0)
		this->scene = std::make_unique<StippleIsoScene>();
	else if (std::strncmp(scene, "cube", 4) == 0)
		this->scene = std::make_unique<CubeScene>();
	else if (std::strncmp(scene, "vector_demo", 11) == 0)
		this->scene = std::make_unique<GlyphScene>();
	else if (std::strncmp(scene, "triangle", 8) == 0)
		this->scene = std::make_unique<TriangleScene>();
	else if (checkExtension(scene, ".tris"))
	{
		if (std::strncmp(scene, "vh:", 3) == 0)
			this->scene = std::make_unique<ClipspaceScene>(scene + 3, ClipspaceScene::ShaderType::VERTEX_HEAVY);
		else if (std::strncmp(scene, "vsh:", 4) == 0)
			this->scene = std::make_unique<ClipspaceScene>(scene + 4, ClipspaceScene::ShaderType::VERTEX_SUPER_HEAVY);
		else if (std::strncmp(scene, "fh:", 3) == 0)
			this->scene = std::make_unique<ClipspaceScene>(scene + 3, ClipspaceScene::ShaderType::FRAGMENT_HEAVY);
		else if (std::strncmp(scene, "fsh:", 4) == 0)
			this->scene = std::make_unique<ClipspaceScene>(scene + 4, ClipspaceScene::ShaderType::FRAGMENT_SUPER_HEAVY);
		else
			this->scene = std::make_unique<ClipspaceScene>(scene);
		//this->scene = std::make_unique<ClipspaceViewScene>(scene);
	}
	else if (checkExtension(scene, ".candy"))
	{
		if (std::strncmp(scene, "vh:", 3) == 0)
			this->scene = std::make_unique<EyeCandyScene>(scene + 3, config, display, EyeCandyScene::ShaderType::VERTEX_HEAVY);
		else if (std::strncmp(scene, "fh:", 3) == 0)
			this->scene = std::make_unique<EyeCandyScene>(scene + 3, config, display, EyeCandyScene::ShaderType::FRAGMENT_HEAVY);
		else
			this->scene = std::make_unique<EyeCandyScene>(scene, config, display);
	}
	else
	{
		if (std::strncmp(scene, "vh:", 3) == 0)
			this->scene = std::make_unique<HeavyScene>(scene + 3, HeavyScene::Type::VERTEX_HEAVY);
		else if (std::strncmp(scene, "fh:", 3) == 0)
			this->scene = std::make_unique<HeavyScene>(scene + 3, HeavyScene::Type::FRAGMENT_HEAVY);
		else
			this->scene = std::make_unique<LoadedScene>(scene);
	}

	if (res_x > 0 && res_y > 0)
		display.resizeWindow(res_x, res_y);

	switchRenderer(config.loadString("current_renderer", "OpenGL"));
}

RenderingSystem::~RenderingSystem()
{
	plugin_man.attach(nullptr);
}

const char* RenderingSystem::rendererName() const
{
	return renderers.empty() ? "" : std::get<0>(renderers[current_plugin]).c_str();
}

void RenderingSystem::releaseRenderer()
{
	display.attach(nullptr, "");
	scene->switchRenderer(nullptr);
	renderer.reset();
}

void RenderingSystem::onRendererPlugInLoaded(const char* name, createRendererFunc create_function)
{
	auto found = std::find_if(begin(renderers), end(renderers), [name](const renderer_list_t::value_type& r) { return std::get<0>(r) == name; });

	if (found != end(renderers))
	{
		if (found - begin(renderers) == current_plugin)
			releaseRenderer();

		std::get<0>(*found) = name;
		std::get<1>(*found) = create_function;
		std::cout << "reloaded renderer " << name << " (" << create_function << ")\n";
	}
	else
	{
		renderers.emplace_back(std::make_tuple(name, create_function));
		std::cout << "loaded renderer " << name << " (" << create_function << ")\n";
	}
}

void RenderingSystem::onRendererPlugInUnloading(createRendererFunc create_function)
{
	auto found = std::find_if(begin(renderers), end(renderers), [create_function](const renderer_list_t::value_type& r) { return std::get<1>(r) == create_function; });

	if (found != end(renderers))
	{
		if (found - begin(renderers) == current_plugin)
			releaseRenderer();
		renderers.erase(found);
		switchRenderer();
	}

	std::cout << "unloaded renderer " << create_function << '\n';
}

void RenderingSystem::switchRenderer(const char* name)
{
	auto found = std::find_if(begin(renderers), end(renderers), [name](const renderer_list_t::value_type& r) { return std::get<0>(r) == name; });

	if (found != end(renderers))
	{
		releaseRenderer();

		current_plugin = found - begin(renderers);
	}
}

void RenderingSystem::switchRenderer(int d)
{
	releaseRenderer();

	const int N = static_cast<int>(renderers.size());

	current_plugin = (((current_plugin + d) % N) + N) % N;
}

void RenderingSystem::render()
{
	if (renderer)
	{
		display.render(*scene, camera);
	}
	else
	{
		if (!renderers.empty())
		{
			renderer = plugin_ptr< ::Renderer>(std::get<1>(renderers[current_plugin])(device, perf_mon));
			scene->switchRenderer(renderer.get());
			display.attach(renderer.get(), std::get<0>(renderers[current_plugin]).c_str());
		}
	}
}

void RenderingSystem::screenshot(const char* filename) const
{
	if (!filename)
	{
		SYSTEMTIME time;
		GetLocalTime(&time);

		std::ostringstream filename;
		filename << time.wYear << '-' << std::setfill('0') << std::setw(2)
		         << time.wMonth << '-' << std::setw(2)
		         << time.wDay << 'T' << std::setw(2)
		         << time.wHour << '_' << std::setw(2)
		         << time.wMinute << '_' << std::setw(2)
		         << time.wSecond << '_' << std::setw(3)
		         << time.wMilliseconds << "_"
		         << rendererName() << ".png";

		PNG::saveRGBA8(filename.str().c_str(), display.screenshot());
		return;
	}

	PNG::saveRGBA8(filename, display.screenshot());
}

void RenderingSystem::attach(const Navigator* navigator)
{
	camera.attach(navigator);
}

void RenderingSystem::attach(GL::platform::MouseInputHandler* mouse_input)
{
	display.attach(mouse_input);
}

void RenderingSystem::attach(GL::platform::KeyboardInputHandler* keyboard_input)
{
	display.attach(keyboard_input);
}

void RenderingSystem::save(Config& config) const
{
	config.saveString("current_renderer", rendererName());
	display.save(config);

	if (scene)
		scene->save(config);
}

void RenderingSystem::sceneButtonPushed(GL::platform::Key c)
{
	scene->handleButton(c);
}
