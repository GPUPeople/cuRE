


#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "RenderingSystem.h"
#include "RecordingPerformanceMonitor.h"


RecordingPerformanceMonitor::RecordingPerformanceMonitor(unsigned int max_num_frames)
	: max_num_frames(max_num_frames),
	  start_time(std::chrono::steady_clock::now())
{
	frame_times.reserve(max_num_frames);
	memory_usage.reserve(max_num_frames);

	auto error_mode = GetErrorMode();
	SetErrorMode(error_mode | SEM_NOGPFAULTERRORBOX);
	SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED);
}

RecordingPerformanceMonitor::~RecordingPerformanceMonitor()
{
	SetThreadExecutionState(ES_CONTINUOUS);
}

void RecordingPerformanceMonitor::reset()
{
}

void RecordingPerformanceMonitor::recordGPUInfo(const char* name, int compute_capability_major, int compute_capability_minor, int num_multiprocessors, int warp_size, int max_threads_per_mp, int regs_per_mp, std::size_t shared_memory_per_mp, std::size_t total_constant_memory, std::size_t total_global_memory, int clock_rate, int max_threads_per_block, int max_regs_per_block, std::size_t max_shared_memory_per_block)
{
	gpu_info = { name, compute_capability_major, compute_capability_minor, num_multiprocessors, warp_size, max_threads_per_mp, regs_per_mp, shared_memory_per_mp, total_constant_memory, total_global_memory, clock_rate, max_threads_per_block, max_regs_per_block, max_shared_memory_per_block };
}

void RecordingPerformanceMonitor::recordDrawingTime(double t)
{
	frame_times.emplace_back(std::chrono::steady_clock::now(), t);

	if (frame_times.size() == max_num_frames)
		PostQuitMessage(0);
}

void RecordingPerformanceMonitor::recordInstrumentationTimers(std::unique_ptr<TimingResult[]> timers, int num_timers, int num_multiprocessors)
{
	instrumentation_data.emplace_back(std::chrono::steady_clock::now(), InstrumentationTimers { num_multiprocessors, num_timers, std::move(timers) });
}

void RecordingPerformanceMonitor::recordMemoryStatus(std::size_t free, std::size_t total)
{
	memory_usage.emplace_back(std::chrono::steady_clock::now(), free, total);
}

void RecordingPerformanceMonitor::recordQueueSize(const std::uint32_t* queue_size, int num_queues)
{
	queue_sizes.emplace_back(std::chrono::steady_clock::now(), std::vector<unsigned int>(queue_size, queue_size + num_queues));
}

void RecordingPerformanceMonitor::recordMaxQueueFillLevel(const std::uint32_t* max_fill_level, int num_queues)
{
	max_queue_fill_levels.emplace_back(std::chrono::steady_clock::now(), std::vector<unsigned int>(max_fill_level, max_fill_level + num_queues));
}


std::ostream& RecordingPerformanceMonitor::printStatus(std::ostream& out) const
{
	return out << "recording...";
}

std::ostream& RecordingPerformanceMonitor::writeData(std::ostream& out) const
{
	using seconds = std::chrono::duration<float>;

	out << "gpu;"
		 << gpu_info.name << ';'
		 << gpu_info.compute_capability_major << '.'
		 << gpu_info.compute_capability_minor << ';'
		 << gpu_info.num_multiprocessors << ';'
		 << gpu_info.warp_size << ';'
		 << gpu_info.max_threads_per_mp << ';'
		 << gpu_info.regs_per_mp << ';'
		 << gpu_info.shared_memory_per_mp << ';'
		 << gpu_info.total_constant_memory << ';'
		 << gpu_info.total_global_memory << ';'
		 << gpu_info.clock_rate << ';'
		 << gpu_info.max_threads_per_block << ';'
		 << gpu_info.max_regs_per_block << ';'
		 << gpu_info.max_shared_memory_per_block << '\n';

	for (const auto& t : frame_times)
		out << "t;" << std::chrono::duration_cast<seconds>(std::get<0>(t) - start_time).count() << ';' << std::get<1>(t) << '\n';

	for (const auto& m : memory_usage)
		out << "mem;" << std::chrono::duration_cast<seconds>(std::get<0>(m) - start_time).count() << ';' << std::get<1>(m) << ';' << std::get<2>(m) << '\n';

	for (const auto& qs : queue_sizes)
	{
		out << "qs;" << std::chrono::duration_cast<seconds>(std::get<0>(qs) - start_time).count();
		for (auto s : std::get<1>(qs))
			out << ';' << s;
		out << '\n';
	}

	for (const auto& ql : max_queue_fill_levels)
	{
		out << "ql;" << std::chrono::duration_cast<seconds>(std::get<0>(ql) - start_time).count();
		for (auto s : std::get<1>(ql))
			out << ';' << s;
		out << '\n';
	}

	for (const auto& d : instrumentation_data)
	{
		auto num_multiprocessors = std::get<1>(d).num_multiprocessors;
		auto num_timers = std::get<1>(d).num_timers;
		auto timers = std::get<1>(d).timers.get();

		for (int i = 0; i < num_multiprocessors; ++i)
		{
			out << "instt;" << std::chrono::duration_cast<seconds>(std::get<0>(d) - start_time).count() << ';' << i;

			for (int j = 0; j < num_timers; ++j)
				out << ';' << timers[i * num_timers + j].t << ';' << timers[i * num_timers + j].N;

			out << '\n';
		}
	}

	return out;
}

void RecordingPerformanceMonitor::saveData(const char* filename, const RenderingSystem* rendering_system) const
{
	if (rendering_system)
	{
		std::ostringstream screenshot_name;
		screenshot_name << filename << ".png";
		rendering_system->screenshot(screenshot_name.str().c_str());
	}

	std::ofstream file(filename);
	writeData(file);
}
