


#ifndef INCLUDED_PERFORMANCE_MONITOR
#define INCLUDED_PERFORMANCE_MONITOR

#pragma once

#include <iosfwd>

#include "Renderer.h"


class PerformanceMonitor : public virtual PerformanceDataCallback
{
protected:
	PerformanceMonitor() = default;
	PerformanceMonitor(const PerformanceMonitor&) = default;
	PerformanceMonitor& operator =(const PerformanceMonitor&) = default;
	~PerformanceMonitor() = default;
public:
	virtual void reset() = 0;

	virtual std::ostream& printStatus(std::ostream& out) const = 0;
};

#endif  // INCLUDED_PERFORMANCE_MONITOR
