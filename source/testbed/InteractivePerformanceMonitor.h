


#ifndef INCLUDED_INTERACTIVE_PERFORMANCE_MONITOR
#define INCLUDED_INTERACTIVE_PERFORMANCE_MONITOR

#pragma once

#include <vector>

#include "PerformanceMonitor.h"


class InteractivePerformanceMonitor : public virtual PerformanceMonitor
{
	double t;

	unsigned int num_frames;

	std::size_t free_memory = 0;
	std::size_t total_memory = 0;

	std::vector<float> queue_size_scale;

	float max_tri_fill_level;
	float min_rast_fill_level;
	float max_rast_fill_level;

public:
	InteractivePerformanceMonitor(const InteractivePerformanceMonitor&) = delete;
	InteractivePerformanceMonitor& operator =(const InteractivePerformanceMonitor&) = delete;

	InteractivePerformanceMonitor();

	void recordGPUInfo(const char* name, int compute_capability_major, int compute_capability_minor, int num_multiprocessors, int warp_size, int max_threads_per_mp, int regs_per_mp, std::size_t shared_memory_per_mp, std::size_t total_constant_memory, std::size_t total_global_memory, int clock_rate, int max_threads_per_block, int max_regs_per_block, std::size_t max_shared_memory_per_block) override;
	void recordDrawingTime(double t) override;
	void recordInstrumentationTimers(std::unique_ptr<TimingResult[]> timers, int num_timers, int num_multiprocessors) override;
	void recordMemoryStatus(std::size_t free, std::size_t total) override;
	void recordQueueSize(const std::uint32_t* queue_size, int num_queues) override;
	void recordMaxQueueFillLevel(const std::uint32_t* max_fill_level, int num_queues) override;

	void reset() override;

	std::ostream& printStatus(std::ostream& out) const override;
};

#endif  // INCLUDED_INTERACTIVE_PERFORMANCE_MONITOR
