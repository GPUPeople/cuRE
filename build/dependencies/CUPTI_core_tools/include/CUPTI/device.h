


#ifndef INCLUDED_CUPTI_DEVICE
#define INCLUDED_CUPTI_DEVICE

#pragma once

#include <cupti.h>


namespace CUPTI
{
	namespace secret
	{
		template <CUpti_DeviceAttribute attribute>
		struct GetDeviceAttributeType;

		template <>
		struct GetDeviceAttributeType<CUPTI_DEVICE_ATTR_MAX_EVENT_ID>
		{
			typedef uint32_t type;
		};

		template <>
		struct GetDeviceAttributeType<CUPTI_DEVICE_ATTR_MAX_EVENT_DOMAIN_ID>
		{
			typedef uint32_t type;
		};

		template <>
		struct GetDeviceAttributeType<CUPTI_DEVICE_ATTR_GLOBAL_MEMORY_BANDWIDTH>
		{
			typedef uint64_t type;
		};

		template <>
		struct GetDeviceAttributeType<CUPTI_DEVICE_ATTR_INSTRUCTION_PER_CYCLE>
		{
			typedef uint32_t type;
		};

		template <>
		struct GetDeviceAttributeType<CUPTI_DEVICE_ATTR_INSTRUCTION_THROUGHPUT_SINGLE_PRECISION>
		{
			typedef uint64_t type;
		};

		template <>
		struct GetDeviceAttributeType<CUPTI_DEVICE_ATTR_MAX_FRAME_BUFFERS>
		{
			typedef uint64_t type;
		};

		template <>
		struct GetDeviceAttributeType<CUPTI_DEVICE_ATTR_PCIE_LINK_RATE>
		{
			typedef uint64_t type;
		};

		template <>
		struct GetDeviceAttributeType<CUPTI_DEVICE_ATTR_PCIE_LINK_WIDTH>
		{
			typedef uint64_t type;
		};

		template <>
		struct GetDeviceAttributeType<CUPTI_DEVICE_ATTR_PCIE_GEN>
		{
			typedef uint64_t type;
		};

		template <>
		struct GetDeviceAttributeType<CUPTI_DEVICE_ATTR_DEVICE_CLASS>
		{
			typedef CUpti_DeviceAttributeDeviceClass type;
		};

		template <typename T>
		T getDeviceAttribute(CUdevice device, CUpti_DeviceAttribute attribute);

#ifndef CUPTI_CORE_TOOLS_DEFINITIONS
		extern template uint32_t getDeviceAttribute<uint32_t>(CUdevice device, CUpti_DeviceAttribute attribute);

		extern template uint64_t getDeviceAttribute<uint64_t>(CUdevice device, CUpti_DeviceAttribute attribute);

		extern template CUpti_DeviceAttributeDeviceClass getDeviceAttribute<CUpti_DeviceAttributeDeviceClass>(CUdevice device, CUpti_DeviceAttribute attribute);
#endif
	}

	template <CUpti_DeviceAttribute attribute>
	inline typename secret::GetDeviceAttributeType<attribute>::type getDeviceAttribute(CUdevice device)
	{
		return secret::getDeviceAttribute<typename secret::GetDeviceAttributeType<attribute>::type>(device, attribute);
	}
}

#endif  // INCLUDED_CUPTI_DEVICE
