


#include "error.h"
#include "device.h"

#define CUPTI_CORE_TOOLS_DEFINITIONS
#include "event.h"


namespace CUPTI
{
	namespace secret
	{
		template <typename T>
		T getEventDomainAttribute(CUpti_EventDomainID event_domain, CUpti_EventDomainAttribute attribute)
		{
			T value;
			size_t size = sizeof(value);
			succeed(cuptiEventDomainGetAttribute(event_domain, attribute, &size, &value));
			return value;
		}

		template <typename T>
		T getEventDomainAttribute(CUdevice device, CUpti_EventDomainID event_domain, CUpti_EventDomainAttribute attribute)
		{
			T value;
			size_t size = sizeof(value);
			succeed(cuptiDeviceGetEventDomainAttribute(device, event_domain, attribute, &size, &value));
			return value;
		}

		template <>
		std::string getEventDomainAttribute<std::string>(CUpti_EventDomainID event_domain, CUpti_EventDomainAttribute attribute)
		{
			std::vector<char> buffer(256);
			while (true)
			{
				size_t buffer_size = buffer.size();
				CUptiResult res = cuptiEventDomainGetAttribute(event_domain, attribute, &buffer_size, &buffer[0]);
				if (res != CUPTI_SUCCESS)
				{
					if (res == CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT)
					{
						buffer.resize(buffer.size() * 2);
						continue;
					}
					else
						throw error(res);
				}
				break;
			}
			return &buffer[0];
		}

		template uint32_t getEventDomainAttribute<uint32_t>(CUdevice device, CUpti_EventDomainID event_domain, CUpti_EventDomainAttribute attribute);

		template CUpti_EventCollectionMethod getEventDomainAttribute<CUpti_EventCollectionMethod>(CUpti_EventDomainID event_domain, CUpti_EventDomainAttribute attribute);

		template <typename T>
		T getEventAttribute(CUpti_EventID event, CUpti_EventAttribute attribute)
		{
			T value;
			size_t size = sizeof(value);
			succeed(cuptiEventGetAttribute(event, attribute, &size, &value));
			return value;
		}

		template CUpti_EventCategory getEventAttribute<CUpti_EventCategory>(CUpti_EventID event, CUpti_EventAttribute attribute);

		template <>
		std::string getEventAttribute<std::string>(CUpti_EventID event, CUpti_EventAttribute attribute)
		{
			std::vector<char> buffer(256);
			while (true)
			{
				size_t buffer_size = buffer.size();
				CUptiResult res = cuptiEventGetAttribute(event, attribute, &buffer_size, &buffer[0]);
				if (res != CUPTI_SUCCESS)
				{
					if (res == CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT)
					{
						buffer.resize(buffer.size() * 2);
						continue;
					}
					else
						throw error(res);
				}
				break;
			}
			return &buffer[0];
		}

		template <typename T>
		T getEventGroupAttribute(CUpti_EventGroup event_group, CUpti_EventGroupAttribute attribute)
		{
			T value;
			size_t size = sizeof(value);
			succeed(cuptiEventGroupGetAttribute(event_group, attribute, &size, &value));
			return value;
		}

		template int getEventGroupAttribute<int>(CUpti_EventGroup event_group, CUpti_EventGroupAttribute attribute);

		template uint32_t getEventGroupAttribute<uint32_t>(CUpti_EventGroup event_group, CUpti_EventGroupAttribute attribute);

		//template CUpti_EventDomainID getEventGroupAttribute<CUpti_EventDomainID>(CUpti_EventGroup event_group, CUpti_EventGroupAttribute attribute);

		template <>
		std::vector<CUpti_EventID> getEventGroupAttribute(CUpti_EventGroup event_group, CUpti_EventGroupAttribute attribute)
		{
			std::vector<CUpti_EventID> buffer(32);
			while (true)
			{
				size_t buffer_size = sizeof(CUpti_EventID) * buffer.size();
				CUptiResult res = cuptiEventGroupGetAttribute(event_group, attribute, &buffer_size, &buffer[0]);
				if (res != CUPTI_SUCCESS)
				{
					if (res == CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT)
					{
						buffer.resize(buffer.size() * 2);
						continue;
					}
					else
						throw error(res);
				}
				break;
			}
			return buffer;
		}
	}

	std::vector<CUpti_EventDomainID> enumEventDomains(CUdevice device)
	{
		auto num_domains = CUPTI::getDeviceAttribute<CUPTI_DEVICE_ATTR_MAX_EVENT_DOMAIN_ID>(device);
		std::vector<CUpti_EventDomainID> domains(num_domains);
		if (num_domains)
		{
			size_t domain_buffer_size = sizeof(CUpti_EventDomainID) * domains.size();
			succeed(cuptiDeviceEnumEventDomains(device, &domain_buffer_size, &domains[0]));
			domains.resize(domain_buffer_size / sizeof(CUpti_EventDomainID));
		}
		return domains;
	}

	uint32_t getNumEventsInDomain(CUpti_EventDomainID domain)
	{
		uint32_t num_events;
		succeed(cuptiEventDomainGetNumEvents(domain, &num_events));
		return num_events;
	}

	std::vector<CUpti_EventID> enumEventsInDomain(CUpti_EventDomainID domain)
	{
		auto num_events = getNumEventsInDomain(domain);
		std::vector<CUpti_EventID> events(num_events);
		if (num_events)
		{
			size_t event_buffer_size = sizeof(CUpti_EventID) * events.size();
			succeed(cuptiEventDomainEnumEvents(domain, &event_buffer_size, &events[0]));
			events.resize(event_buffer_size / sizeof(CUpti_EventID));
		}
		return events;
	}

	CUpti_EventID getEventID(CUdevice device, const char* name)
	{
		CUpti_EventID event_id;
		succeed(cuptiEventGetIdFromName(device, name, &event_id));
		return event_id;
	}

	EventGroup createEventGroup(CUcontext context)
	{
		CUpti_EventGroup event_group;
		succeed(cuptiEventGroupCreate(context, &event_group, 0U));
		return event_group;
	}
}
