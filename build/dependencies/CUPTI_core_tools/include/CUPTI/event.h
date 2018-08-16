


#ifndef INCLUDED_CUPTI_EVENT
#define INCLUDED_CUPTI_EVENT

#pragma once

#include <utility>
#include <vector>
#include <string>

#include <cupti.h>

#include "error.h"


namespace CUPTI
{
	namespace secret
	{
		template <CUpti_EventDomainAttribute attribute>
		struct GetEventDomainAttributeType;

		template <>
		struct GetEventDomainAttributeType<CUPTI_EVENT_DOMAIN_ATTR_NAME>
		{
			typedef std::string type;
		};

		template <>
		struct GetEventDomainAttributeType<CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT>
		{
			typedef uint32_t type;
		};

		template <>
		struct GetEventDomainAttributeType<CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT>
		{
			typedef uint32_t type;
		};

		template <>
		struct GetEventDomainAttributeType<CUPTI_EVENT_DOMAIN_ATTR_COLLECTION_METHOD>
		{
			typedef CUpti_EventCollectionMethod type;
		};

		template <typename T>
		T getEventDomainAttribute(CUpti_EventDomainID event_domain, CUpti_EventDomainAttribute attribute);

		template <typename T>
		T getEventDomainAttribute(CUdevice device, CUpti_EventDomainID event_domain, CUpti_EventDomainAttribute attribute);

		template <CUpti_EventAttribute attribute>
		struct GetEventAttributeType;

		template <>
		struct GetEventAttributeType<CUPTI_EVENT_ATTR_NAME>
		{
			typedef std::string type;
		};

		template <>
		struct GetEventAttributeType<CUPTI_EVENT_ATTR_SHORT_DESCRIPTION>
		{
			typedef std::string type;
		};

		template <>
		struct GetEventAttributeType<CUPTI_EVENT_ATTR_LONG_DESCRIPTION>
		{
			typedef std::string type;
		};

		template <>
		struct GetEventAttributeType<CUPTI_EVENT_ATTR_CATEGORY>
		{
			typedef CUpti_EventCategory type;
		};

		template <typename T>
		T getEventAttribute(CUpti_EventID event, CUpti_EventAttribute attribute);

		template <CUpti_EventGroupAttribute>
		struct GetEventGroupAttributeType;

		template <>
		struct GetEventGroupAttributeType<CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID>
		{
			typedef CUpti_EventDomainID type;
		};

		template <>
		struct GetEventGroupAttributeType<CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES>
		{
			typedef int type;
		};

		template <>
		struct GetEventGroupAttributeType<CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS>
		{
			typedef uint32_t type;
		};

		template <>
		struct GetEventGroupAttributeType<CUPTI_EVENT_GROUP_ATTR_EVENTS>
		{
			typedef std::vector<CUpti_EventID> type;
		};

		template <>
		struct GetEventGroupAttributeType<CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT>
		{
			typedef uint32_t type;
		};

		template <typename T>
		T getEventGroupAttribute(CUpti_EventGroup event_group, CUpti_EventGroupAttribute attribute);

		template <>
		std::string getEventDomainAttribute<std::string>(CUpti_EventDomainID event_domain, CUpti_EventDomainAttribute attribute);

		template <>
		std::vector<CUpti_EventID> getEventGroupAttribute<std::vector<CUpti_EventID> >(CUpti_EventGroup event_group, CUpti_EventGroupAttribute attribute);

#ifndef CUPTI_CORE_TOOLS_DEFINITIONS
		extern template uint32_t getEventDomainAttribute<uint32_t>(CUdevice device, CUpti_EventDomainID event_domain, CUpti_EventDomainAttribute attribute);

		extern template CUpti_EventCollectionMethod getEventDomainAttribute<CUpti_EventCollectionMethod>(CUpti_EventDomainID event_domain, CUpti_EventDomainAttribute attribute);

		extern template std::string getEventAttribute<std::string>(CUpti_EventID event, CUpti_EventAttribute attribute);

		extern template CUpti_EventCategory getEventAttribute<CUpti_EventCategory>(CUpti_EventID event, CUpti_EventAttribute attribute);

		extern template int getEventGroupAttribute<int>(CUpti_EventGroup event_group, CUpti_EventGroupAttribute attribute);

		extern template uint32_t getEventGroupAttribute<uint32_t>(CUpti_EventGroup event_group, CUpti_EventGroupAttribute attribute);

		//extern template CUpti_EventDomainID getEventGroupAttribute<CUpti_EventDomainID>(CUpti_EventGroup event_group, CUpti_EventGroupAttribute attribute);
#endif
	}

	template <CUpti_EventDomainAttribute attribute>
	inline typename secret::GetEventDomainAttributeType<attribute>::type getEventDomainAttribute(CUpti_EventDomainID event_domain)
	{
		return secret::getEventDomainAttribute<typename secret::GetEventDomainAttributeType<attribute>::type>(event_domain, attribute);
	}

	template <>
	typename secret::GetEventDomainAttributeType<CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT>::type getEventDomainAttribute<CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT>(CUpti_EventDomainID event_domain);

	template <>
	typename secret::GetEventDomainAttributeType<CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT>::type getEventDomainAttribute<CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT>(CUpti_EventDomainID event_domain);

	template <CUpti_EventDomainAttribute attribute>
	inline typename secret::GetEventDomainAttributeType<attribute>::type getEventDomainAttribute(CUdevice device, CUpti_EventDomainID event_domain)
	{
		return secret::getEventDomainAttribute<typename secret::GetEventDomainAttributeType<attribute>::type>(device, event_domain, attribute);
	}

	template <CUpti_EventAttribute attribute>
	inline typename secret::GetEventAttributeType<attribute>::type getEventAttribute(CUpti_EventID event)
	{
		return secret::getEventAttribute<typename secret::GetEventAttributeType<attribute>::type>(event, attribute);
	}

	template <CUpti_EventGroupAttribute attribute>
	inline typename secret::GetEventGroupAttributeType<attribute>::type getEventGroupAttribute(CUpti_EventGroup event_group)
	{
		return secret::getEventGroupAttribute<typename secret::GetEventGroupAttributeType<attribute>::type>(event_group, attribute);
	}

	template <CUpti_EventGroupAttribute attribute>
	void setEventGroupAttribute(CUpti_EventGroup event_group, typename secret::GetEventGroupAttributeType<attribute>::type value);

	template <>
	inline void setEventGroupAttribute<CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES>(CUpti_EventGroup event_group, int value)
	{
		succeed(cuptiEventGroupSetAttribute(event_group, CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(int), &value));
	}

	std::vector<CUpti_EventDomainID> enumEventDomains(CUdevice device);

	uint32_t getNumEventsInDomain(CUpti_EventDomainID domain);
	std::vector<CUpti_EventID> enumEventsInDomain(CUpti_EventDomainID domain);

	CUpti_EventID getEventID(CUdevice device, const char* name);

	class EventGroup
	{
	private:
		EventGroup(const EventGroup&);
		EventGroup& operator=(const EventGroup&);

		CUpti_EventGroup event_group;

	public:
		EventGroup(CUpti_EventGroup event_group = nullptr)
		    : event_group(event_group)
		{
		}

		~EventGroup()
		{
			if (event_group)
			{
				cuptiEventGroupDisable(event_group);
				cuptiEventGroupDestroy(event_group);
			}
		}

		EventGroup(EventGroup&& g)
		    : event_group(g.event_group)
		{
			g.event_group = nullptr;
		}

		EventGroup& operator=(EventGroup&& g)
		{
			using std::swap;
			swap(event_group, g.event_group);
			return *this;
		}

		operator CUpti_EventGroup() const { return event_group; }
	};

	EventGroup createEventGroup(CUcontext context);
}

#endif  // INCLUDED_CUPTI_EVENT
