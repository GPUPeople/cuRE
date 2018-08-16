


#ifndef INCLUDED_RECORDING_PERFORMANCE_MONITOR
#define INCLUDED_RECORDING_PERFORMANCE_MONITOR

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <chrono>

#include "PerformanceMonitor.h"

class RenderingSystem;


class RecordingPerformanceMonitor : public virtual PerformanceMonitor
{
	const unsigned int max_num_frames;

	std::vector<std::tuple<std::chrono::steady_clock::time_point, double>> frame_times;
	std::vector<std::tuple<std::chrono::steady_clock::time_point, std::size_t, std::size_t>> memory_usage;
	std::vector<std::tuple<std::chrono::steady_clock::time_point, std::vector<unsigned int>>> queue_sizes;
	std::vector<std::tuple<std::chrono::steady_clock::time_point, std::vector<unsigned int>>> max_queue_fill_levels;

	struct GPUInfo
	{
		std::string name;
		int compute_capability_major;
		int compute_capability_minor;
		int num_multiprocessors;
		int warp_size;
		int max_threads_per_mp;
		int regs_per_mp;
		std::size_t shared_memory_per_mp;
		std::size_t total_constant_memory;
		std::size_t total_global_memory;
		int clock_rate;
		int max_threads_per_block;
		int max_regs_per_block;
		std::size_t max_shared_memory_per_block;
	} gpu_info;

	struct InstrumentationTimers
	{
		int num_multiprocessors;
		int num_timers;
		std::unique_ptr<TimingResult[]> timers;
	};

	std::vector<std::tuple<std::chrono::steady_clock::time_point, InstrumentationTimers>> instrumentation_data;

	std::chrono::steady_clock::time_point start_time;

public:
	RecordingPerformanceMonitor(const RecordingPerformanceMonitor&) = delete;
	RecordingPerformanceMonitor& operator =(const RecordingPerformanceMonitor&) = delete;

	RecordingPerformanceMonitor(unsigned int max_num_frames);
	~RecordingPerformanceMonitor();

	void recordGPUInfo(const char* name, int compute_capability_major, int compute_capability_minor, int num_multiprocessors, int warp_size, int max_threads_per_mp, int regs_per_mp, std::size_t shared_memory_per_mp, std::size_t total_constant_memory, std::size_t total_global_memory, int clock_rate, int max_threads_per_block, int max_regs_per_block, std::size_t max_shared_memory_per_block) override;
	void recordDrawingTime(double t) override;
	void recordInstrumentationTimers(std::unique_ptr<TimingResult[]> timers, int num_timers, int num_multiprocessors) override;
	void recordMemoryStatus(std::size_t free, std::size_t total) override;
	void recordQueueSize(const std::uint32_t* queue_size, int num_queues) override;
	void recordMaxQueueFillLevel(const std::uint32_t* max_fill_level, int num_queues) override;

	void reset() override;

	std::ostream& printStatus(std::ostream& out) const override;

	std::ostream& writeData(std::ostream& out) const;

	void saveData(const char* filename, const RenderingSystem* rendering_system = nullptr) const;
};

#endif  // INCLUDED_RECORDING_PERFORMANCE_MONITOR
