


#include "error.h"

#define CUPTI_CORE_TOOLS_DEFINITIONS
#include "device.h"


namespace CUPTI
{
	namespace secret
	{
		template <typename T>
		T getDeviceAttribute(CUdevice device, CUpti_DeviceAttribute attribute)
		{
			T value;
			size_t size = sizeof(value);
			succeed(cuptiDeviceGetAttribute(device, attribute, &size, &value));
			return value;
		}

		template uint32_t getDeviceAttribute<uint32_t>(CUdevice device, CUpti_DeviceAttribute attribute);

		template uint64_t getDeviceAttribute<uint64_t>(CUdevice device, CUpti_DeviceAttribute attribute);

		template CUpti_DeviceAttributeDeviceClass getDeviceAttribute<CUpti_DeviceAttributeDeviceClass>(CUdevice device, CUpti_DeviceAttribute attribute);
	}
}
