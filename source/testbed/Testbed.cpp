


#include <iostream>

#include <CUDA/device.h>
#include <CUDA/error.h>

#include <GL/platform/Application.h>

#include "Config.h"
#include "ConsoleHandler.h"
#include "FirstPersonNavigator.h"
#include "InputHandler.h"
#include "OrbitalNavigator.h"
#include "PlugInManager.h"

#include "InteractivePerformanceMonitor.h"
#include "RecordingPerformanceMonitor.h"

#include "RenderingSystem.h"

#include "Testbed.h"


Testbed::Testbed()
{
	static struct init_CUDA_t
	{
		init_CUDA_t()
		{
			succeed(cuInit(0U));
		}
	} cuda_init;
}

std::ostream& Testbed::listDevices(std::ostream& out) const
{
	for (int i = 0; i < CU::getDeviceCount(); ++i)
	{
		auto device = CU::getDevice(i);
		out << i << '\t' << CU::getDeviceName(device) << '\n';
	}

	return out;
}

int Testbed::run(Config& config, int res_x, int res_y, int record, const char* perf_file)
{
	PlugInManager plugin_man(config);

	CUdevice device = CU::getDevice(config.loadInt("device", 0));
	const char* scenefile = config.loadString("scene", "icosahedron");

	union PerfMonitors {
		PerfMonitors() {}
		~PerfMonitors() {}
		InteractivePerformanceMonitor interactive_mon;
		RecordingPerformanceMonitor recording_mon;
	} perf_monitors;

	PerformanceMonitor* perf_mon = record > 0 ? static_cast<PerformanceMonitor*>(new (&perf_monitors) RecordingPerformanceMonitor(record)) : static_cast<PerformanceMonitor*>(new (&perf_monitors) InteractivePerformanceMonitor());

	RenderingSystem rendering_system(plugin_man, config, device, perf_mon, scenefile, res_x, res_y);

	OrbitalNavigator navigator(config.loadConfig("orbital_navigator"), -math::constants<float>::pi() * 0.5f, 0.0f, 10.0f);
	//FirstPersonNavigator navigator(1.0f, 0.1f, 0.0f);

	ConsoleHandler console_handler(navigator, rendering_system, plugin_man);

	InputHandler input_handler(navigator, rendering_system, plugin_man);

	rendering_system.attach(static_cast<GL::platform::KeyboardInputHandler*>(&input_handler));
	rendering_system.attach(static_cast<GL::platform::MouseInputHandler*>(&input_handler));
	rendering_system.attach(&navigator);

	GL::platform::run(rendering_system); //, &console_handler);

	if (record > 0)
		perf_monitors.recording_mon.saveData(perf_file, &rendering_system);

	navigator.save(config.loadConfig("orbital_navigator"));
	rendering_system.save(config);
	plugin_man.save(config);

	return 0;
}
