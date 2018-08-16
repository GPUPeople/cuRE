


#include "error.h"

#define CUPTI_CORE_TOOLS_DEFINITIONS
#include "metric.h"


namespace CUPTI
{
	namespace secret
	{
		template <typename T>
		T getMetricAttribute(CUpti_MetricID metric, CUpti_MetricAttribute attribute)
		{
			T value;
			size_t size = sizeof(value);
			succeed(cuptiMetricGetAttribute(metric, attribute, &size, &value));
			return value;
		}

		template CUpti_MetricCategory getMetricAttribute<CUpti_MetricCategory>(CUpti_MetricID metric, CUpti_MetricAttribute attribute);

		template CUpti_MetricValueKind getMetricAttribute<CUpti_MetricValueKind>(CUpti_MetricID metric, CUpti_MetricAttribute attribute);

		template CUpti_MetricEvaluationMode getMetricAttribute<CUpti_MetricEvaluationMode>(CUpti_MetricID metric, CUpti_MetricAttribute attribute);

		template <>
		std::string getMetricAttribute<std::string>(CUpti_MetricID metric, CUpti_MetricAttribute attribute)
		{
			std::vector<char> buffer(256);
			while (true)
			{
				size_t buffer_size = buffer.size();
				CUptiResult res = cuptiMetricGetAttribute(metric, attribute, &buffer_size, &buffer[0]);
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
	}

	uint32_t getNumMetrics()
	{
		uint32_t num_metrics;
		succeed(cuptiGetNumMetrics(&num_metrics));
		return num_metrics;
	}

	uint32_t getNumMetrics(CUdevice device)
	{
		uint32_t num_metrics;
		succeed(cuptiDeviceGetNumMetrics(device, &num_metrics));
		return num_metrics;
	}

	std::vector<CUpti_MetricID> enumMetrics()
	{
		auto num_metrics = getNumMetrics();
		std::vector<CUpti_MetricID> metrics(num_metrics);
		if (num_metrics)
		{
			size_t metric_buffer_size = sizeof(CUpti_MetricID) * metrics.size();
			succeed(cuptiEnumMetrics(&metric_buffer_size, &metrics[0]));
			metrics.resize(metric_buffer_size / sizeof(CUpti_MetricID));
		}
		return metrics;
	}

	std::vector<CUpti_MetricID> enumMetrics(CUdevice device)
	{
		auto num_metrics = getNumMetrics(device);
		std::vector<CUpti_MetricID> metrics(num_metrics);
		if (num_metrics)
		{
			size_t metric_buffer_size = sizeof(CUpti_MetricID) * metrics.size();
			succeed(cuptiDeviceEnumMetrics(device, &metric_buffer_size, &metrics[0]));
			metrics.resize(metric_buffer_size / sizeof(CUpti_MetricID));
		}
		return metrics;
	}

	CUpti_MetricID getMetricID(CUdevice device, const char* name)
	{
		CUpti_MetricID metric_id;
		succeed(cuptiMetricGetIdFromName(device, name, &metric_id));
		return metric_id;
	}

	uint32_t getNumEventsInMetric(CUpti_MetricID metric)
	{
		uint32_t num_events;
		succeed(cuptiMetricGetNumEvents(metric, &num_events));
		return num_events;
	}

	std::vector<CUpti_EventID> enumEventsInMetric(CUpti_MetricID metric)
	{
		auto num_events = getNumEventsInMetric(metric);
		std::vector<CUpti_EventID> events(num_events);
		if (num_events)
		{
			size_t event_buffer_size = sizeof(CUpti_EventID) * events.size();
			succeed(cuptiMetricEnumEvents(metric, &event_buffer_size, &events[0]));
			events.resize(event_buffer_size / sizeof(CUpti_EventID));
		}
		return events;
	}

	uint32_t getNumProperties(CUpti_MetricID metric)
	{
		uint32_t num_properties;
		succeed(cuptiMetricGetNumProperties(metric, &num_properties));
		return num_properties;
	}

	std::vector<CUpti_MetricPropertyID> enumProperties(CUpti_MetricID metric)
	{
		auto num_properties = getNumProperties(metric);
		std::vector<CUpti_MetricPropertyID> properties(num_properties);
		if (num_properties)
		{
			size_t property_buffer_size = sizeof(CUpti_MetricPropertyID) * properties.size();
			succeed(cuptiMetricEnumProperties(metric, &property_buffer_size, &properties[0]));
			properties.resize(property_buffer_size / sizeof(CUpti_MetricPropertyID));
		}
		return properties;
	}
}
