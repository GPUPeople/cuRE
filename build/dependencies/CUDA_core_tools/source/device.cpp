


#include <stdexcept>

#include <CUDA/error.h>
#include <CUDA/device.h>


namespace CU
{
	CUdevice getDevice(int ordinal)
	{
		CUdevice device;
		succeed(cuDeviceGet(&device, ordinal));
		return device;
	}

	CUdevice getDevice(CUcontext ctx)
	{
		CUdevice device;
		succeed(cuCtxGetDevice(&device));
		return device;
	}

	int getDeviceCount()
	{
		int num;
		succeed(cuDeviceGetCount(&num));
		return num;
	}

	std::string getDeviceName(CUdevice device)
	{
		char name[512];
		succeed(cuDeviceGetName(name, 512, device));
		return std::string(name);
	}

	std::size_t getDeviceMemory(CUdevice device)
	{
		std::size_t mem;
		succeed(cuDeviceTotalMem(&mem, device));
		return mem;
	}

	CUdevice findMatchingDevice(int cc_major, int cc_minor)
	{
		int num_devices = getDeviceCount();
		for (int i = 0; i < num_devices; ++i)
		{
			CUdevice dev = getDevice(i);
			int major, minor;
			succeed(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
			succeed(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
			if (cc_major == major && cc_minor == minor)
				return dev;
		}
		throw std::runtime_error("No device of matching CC found");
	}
}
