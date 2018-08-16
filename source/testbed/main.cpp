

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "argparse.h"

#include "Testbed.h"

using namespace std::literals;


namespace
{
	std::ostream& printUsage(std::ostream& file)
	{
		return file << R"""(available options:
  --device <number>    CUDA device to use (default: 0)
  --scene <name>       scene to load (default: 'Icosahedron')
  --config <path>      config file to use (default: 'config.cfg')
  --renderer <name>    force specific renderer module
  --res_x <number>     force specific resolution
  --res_y <number>     force specific resolution
  --record <number>    number of frames to record; starts testbed in batch mode
  --perf-file <path>   output performance data to this file (default: 'perf.csv')
  --list-devices       list all available GPUs
   -h --help           display this message
)""";
	}
}

int main(int argc, char* argv[])
{
	try
	{
		Testbed testbed;

		int device = -1;
		const char* scene = nullptr;
		const char* config_file = "config.cfg";
		const char* renderer = nullptr;
		int res_x = -1;
		int res_y = -1;
		int record = 0;
		const char* perf_file = "perf.csv";

		for (const char* const* a = argv + 1; *a; ++a)
		{
			if (!parseIntArgument(device, a, "--device"sv))
			if (!parseStringArgument(scene, a, "--scene"sv))
			if (!parseStringArgument(config_file, a, "--config"sv))
			if (!parseStringArgument(renderer, a, "--renderer"sv))
			if (!parseIntArgument(res_x, a, "--res-x"sv))
			if (!parseIntArgument(res_y, a, "--res-y"sv))
			if (!parseIntArgument(record, a, "--record"sv))
			if (!parseStringArgument(perf_file, a, "--perf-file"sv))
			if (parseBoolFlag(a, "--list-devices"sv))
			{
				testbed.listDevices(std::cout) << std::endl;
				return 0;
			}
			else if (parseBoolFlag(a, "-h"sv) || parseBoolFlag(a, "--help"sv))
			{
				printUsage(std::cout) << std::endl;
				return 0;
			}
			else
				throw usage_error("unknown argument");
		}

		Config config = loadConfig(config_file);

		if (device >= 0)
			config.saveInt("device", device);

		if (scene)
			config.saveString("scene", scene);

		if (renderer)
		{
			config.saveTuple("modules", { renderer });
			config.saveString("current_renderer", "");
		}

		testbed.run(config, res_x, res_y, record, perf_file);

		save(config, config_file);
	}
	catch (const usage_error& e)
	{
		std::cerr << "error: " << e.what() << std::endl;
		printUsage(std::cout);
		return -1;
	}
	catch (const std::exception& e)
	{
		std::cerr << "error: " << e.what() << std::endl;
		return -1;
	}
	catch (...)
	{
		std::cerr << "unknown exception" << std::endl;
		return -128;
	}

	return 0;
}
