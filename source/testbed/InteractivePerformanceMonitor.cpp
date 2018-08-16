


#include <limits>
#include <algorithm>
#include <iostream>
#include <iomanip>

#include "InteractivePerformanceMonitor.h"


InteractivePerformanceMonitor::InteractivePerformanceMonitor()
{
	reset();
}


void InteractivePerformanceMonitor::reset()
{
	t = 0.0;
	num_frames = 0U;
	max_tri_fill_level = 0.0f;
	min_rast_fill_level = std::numeric_limits<float>::max();
	max_rast_fill_level = 0.0f;
}

void InteractivePerformanceMonitor::recordGPUInfo(const char* name, int compute_capability_major, int compute_capability_minor, int num_multiprocessors, int warp_size, int max_threads_per_mp, int regs_per_mp, std::size_t shared_memory_per_mp, std::size_t total_constant_memory, std::size_t total_global_memory, int clock_rate, int max_threads_per_block, int max_regs_per_block, std::size_t max_shared_memory_per_block)
{
}

void InteractivePerformanceMonitor::recordDrawingTime(double t)
{
	this->t += t;
	++num_frames;
}

void InteractivePerformanceMonitor::recordInstrumentationTimers(std::unique_ptr<TimingResult[]> timers, int num_timers, int num_multiprocessors)
{
}

void InteractivePerformanceMonitor::recordMemoryStatus(std::size_t free, std::size_t total)
{
	free_memory = free;
	total_memory = total;
}

void InteractivePerformanceMonitor::recordQueueSize(const std::uint32_t* queue_size, int num_queues)
{
	queue_size_scale.resize(num_queues);
	for (int i = 0; i < num_queues; ++i)
		queue_size_scale[i] = 1.0f / queue_size[i];
}

void InteractivePerformanceMonitor::recordMaxQueueFillLevel(const std::uint32_t* max_fill_level, int num_queues)
{
	float tri_level = max_fill_level[0] * queue_size_scale[0];
	max_tri_fill_level = std::max(tri_level, max_tri_fill_level);

	for (int i = 1; i < num_queues; ++i)
	{
		float level = max_fill_level[i] * queue_size_scale[i];
		min_rast_fill_level = std::min(level, min_rast_fill_level);
		max_rast_fill_level = std::max(level, max_rast_fill_level);
	}
}


std::ostream& InteractivePerformanceMonitor::printStatus(std::ostream& out) const
{
	double spf = t / num_frames;
	out << std::fixed << "      t = " << std::setprecision(1) << spf * 1000.0 << " ms  (" << 1.0 / spf << " fps)    memory usage: " << std::setprecision(2) << (total_memory - free_memory) * 100.0f / total_memory << '%';
	if (!queue_size_scale.empty())
	{
		out << "    tribuffer: " << max_tri_fill_level * 100.0f << "%  q_min: " << min_rast_fill_level * 100.0f << "%  q_max: " << max_rast_fill_level * 100.0f << '%';
	}
	return out;
}
