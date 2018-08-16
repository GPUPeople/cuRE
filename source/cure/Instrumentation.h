


#ifndef INCLUDED_CURE_INSTRUMENTATION
#define INCLUDED_CURE_INSTRUMENTATION

#pragma once

#include <cstdint>
#include <vector>

#include <CUDA/module.h>
#include <CUDA/memory.h>

#include <CUPTI/event.h>

#include <Renderer.h>
#include "pipeline/config.h"


struct PerformanceDataCallback;

namespace cuRE
{
	class PipelineModule;

	class Instrumentation
	{
		static const int NUM_PER_BLOCK_TIMERS = Timers::NUM_TIMERS;

		CUcontext context;
		CUdevice device;

		CUfunction reset_kernel;
		CUfunction read_data_kernel;

		CUdeviceptr mp_index_buffer;
		CU::unique_ptr per_block_timing_buffer;

		CUdeviceptr triangle_buffer;
		CUdeviceptr rasterizer_queues;

		//unsigned int blocks_recorded = 0U;

		//CUPTI::EventGroup prof_counter_events;
		//size_t num_prof_counter_events;
		//size_t num_prof_counter_instances;

		std::vector<std::uint8_t> mp_indices;
		std::vector<TimingResult> per_block_timings[NUM_PER_BLOCK_TIMERS];
		std::vector<std::uint32_t> max_queue_fill_levels;

		int num_multiprocessors;
		std::unique_ptr<TimingResult[]> results;

		void reportMaxQueueFillLevel(PerformanceDataCallback& perf_mon) const;

		void reset();

	public:
		Instrumentation(const Instrumentation&) = delete;
		Instrumentation& operator =(const Instrumentation&) = delete;

		Instrumentation(const PipelineModule& module);

		void record(unsigned int num_blocks_launched);

		void reportQueueSizes(PerformanceDataCallback& perf_mon) const;
		void reportTelemetry(PerformanceDataCallback& perf_mon);
	};
}

#endif  // INCLUDED_CURE_INSTRUMENTATION
