


#ifndef INCLUDED_CUPTI_METRIC
#define INCLUDED_CUPTI_METRIC

#pragma once

#include <vector>
#include <string>

#include <cupti.h>


namespace CUPTI
{
	namespace secret
	{
		template <CUpti_MetricAttribute attribute>
		struct GetMetricAttributeType;

		template <>
		struct GetMetricAttributeType<CUPTI_METRIC_ATTR_NAME>
		{
			typedef std::string type;
		};

		template <>
		struct GetMetricAttributeType<CUPTI_METRIC_ATTR_SHORT_DESCRIPTION>
		{
			typedef std::string type;
		};

		template <>
		struct GetMetricAttributeType<CUPTI_METRIC_ATTR_LONG_DESCRIPTION>
		{
			typedef std::string type;
		};

		template <>
		struct GetMetricAttributeType<CUPTI_METRIC_ATTR_CATEGORY>
		{
			typedef CUpti_MetricCategory type;
		};

		template <>
		struct GetMetricAttributeType<CUPTI_METRIC_ATTR_VALUE_KIND>
		{
			typedef CUpti_MetricValueKind type;
		};

		template <>
		struct GetMetricAttributeType<CUPTI_METRIC_ATTR_EVALUATION_MODE>
		{
			typedef CUpti_MetricEvaluationMode type;
		};

		template <typename T>
		T getMetricAttribute(CUpti_MetricID metric, CUpti_MetricAttribute attribute);

		template <>
		std::string getMetricAttribute<std::string>(CUpti_MetricID metric, CUpti_MetricAttribute attribute);

#ifndef CUPTI_CORE_TOOLS_DEFINITIONS
		extern template CUpti_MetricCategory getMetricAttribute<CUpti_MetricCategory>(CUpti_MetricID metric, CUpti_MetricAttribute attribute);

		extern template CUpti_MetricValueKind getMetricAttribute<CUpti_MetricValueKind>(CUpti_MetricID metric, CUpti_MetricAttribute attribute);

		extern template CUpti_MetricEvaluationMode getMetricAttribute<CUpti_MetricEvaluationMode>(CUpti_MetricID metric, CUpti_MetricAttribute attribute);
#endif
	}

	template <CUpti_MetricAttribute attribute>
	inline typename secret::GetMetricAttributeType<attribute>::type getMetricAttribute(CUpti_MetricID metric)
	{
		return secret::getMetricAttribute<typename secret::GetMetricAttributeType<attribute>::type>(metric, attribute);
	}

	uint32_t getNumMetrics();
	uint32_t getNumMetrics(CUdevice device);

	std::vector<CUpti_MetricID> enumMetrics();

	std::vector<CUpti_MetricID> enumMetrics(CUdevice device);

	CUpti_MetricID getMetricID(CUdevice device, const char* name);

	uint32_t getNumEventsInMetric(CUpti_MetricID metric);
	std::vector<CUpti_EventID> enumEventsInMetric(CUpti_MetricID metric);

	uint32_t getNumProperties(CUpti_MetricID metric);
	std::vector<CUpti_MetricPropertyID> enumProperties(CUpti_MetricID metric);
}

#endif  // INCLUDED_CUPTI_METRIC
