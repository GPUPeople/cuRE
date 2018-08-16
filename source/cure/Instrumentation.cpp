


#include <cstdint>
//#include <cstring>
#include <type_traits>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <CUDA/context.h>
#include <CUDA/device.h>

#include <utils.h>

#include <Renderer.h>

#include "pipeline/config.h"

#include "PipelineModule.h"
#include "Instrumentation.h"


namespace
{
	std::ostream& operator <<(std::ostream& out, TILE_ACCESS_MODE mode)
	{
		switch (mode)
		{
		case TILE_ACCESS_MODE::WARP_EXCLUSIVE:
			return out << "WARP_EXCLUSIVE";
		case TILE_ACCESS_MODE::WARP_PER_FRAGMENT:
			return out << "WARP_PER_FRAGMENT";
		case TILE_ACCESS_MODE::COVERAGE_MASK:
			return out << "COVERAGE_MASK";
		}

		throw std::runtime_error("invalid enum value");
	}

	std::ostream& operator <<(std::ostream& out, FRAMEBUFFER_SYNC_METHOD mode)
	{
		switch (mode)
		{
		case FRAMEBUFFER_SYNC_METHOD::NO_SYNC:
			return out << "NO_SYNC";
		case FRAMEBUFFER_SYNC_METHOD::SYNC_ALL:
			return out << "SYNC_ALL";
		case FRAMEBUFFER_SYNC_METHOD::SYNC_FRAGMENTS:
			return out << "SYNC_FRAGMENTS";
		case FRAMEBUFFER_SYNC_METHOD::SUBTILE_SYNC:
			return out << "SUBTILE_SYNC";
		case FRAMEBUFFER_SYNC_METHOD::MASK_SYNC:
			return out << "MASK_SYNC";
		case FRAMEBUFFER_SYNC_METHOD::POLLED_MASK_SYNC:
			return out << "POLLED_MASK_SYNC";
		}

		throw std::runtime_error("invalid enum value");
	}

	std::ostream& operator <<(std::ostream& out, PATTERNTECHNIQUE mode)
	{
		switch (mode)
		{
		case PATTERNTECHNIQUE::DIAGONAL:
			return out << "DIAGONAL";
		case PATTERNTECHNIQUE::OFFSET:
			return out << "OFFSET";
		case PATTERNTECHNIQUE::OFFSET_SHIFT:
			return out << "OFFSET_SHIFT";
		case PATTERNTECHNIQUE::OFFSET_SHIFT_SLIM:
			return out << "OFFSET_SHIFT_SLIM";
		case PATTERNTECHNIQUE::DIAGONAL_ITERATIVE:
			return out << "DIAGONAL_ITERATIVE";
		case PATTERNTECHNIQUE::OFFSET_ITERATIVE:
			return out << "OFFSET_ITERATIVE";
		case PATTERNTECHNIQUE::OFFSET_SHIFT_ITERATIVE:
			return out << "OFFSET_SHIFT_ITERATIVE";
		}

		throw std::runtime_error("invalid enum value");
	}

	std::ostream& identify(std::ostream& out)
	{
		out << "module identification:\n"
		    << "\tMAX_NUM_RASTERIZERS: " << MAX_NUM_RASTERIZERS << '\n'
		    << "\tMAX_NUM_BLOCKS: " << MAX_NUM_BLOCKS << '\n'
		    << "\tMAX_WARPS_PER_BLOCK: " << MAX_WARPS_PER_BLOCK << '\n' << '\n'

		    << "\tNUM_INTERPOLATORS: " << NUM_INTERPOLATORS << '\n' << '\n'

		    << "\tCLIPPING: " << CLIPPING << '\n'
		    << "\tBACKFACE_CULLING: " << BACKFACE_CULLING << '\n'
		    << "\tDEPTH_TEST: " << DEPTH_TEST << '\n'
		    << "\tDEPTH_WRITE: " << DEPTH_WRITE << '\n'
		    << "\tBLENDING: " << BLENDING << '\n'
		    << "\tFRAMEBUFFER_SRGB: " << FRAMEBUFFER_SRGB << '\n' << '\n'

		    << "\tWIREFRAME: " << WIREFRAME << '\n'
		    << "\tDRAW_BOUNDING_BOX: " << DRAW_BOUNDING_BOX << '\n' << '\n'

		    << "\tTRIANGLE_BUFFER_SIZE: " << TRIANGLE_BUFFER_SIZE << '\n'
		    << "\tRASTERIZER_QUEUE_SIZE: " << RASTERIZER_QUEUE_SIZE << '\n'
		    << "\tRASTERIZATION_CONSUME_THRESHOLD: " << RASTERIZATION_CONSUME_THRESHOLD << '\n'
		    << "\tDYNAMIC_RASTERIZER_EFFICIENT_THRESHOLD: " << DYNAMIC_RASTERIZER_EFFICIENT_THRESHOLD << '\n' << '\n'

		    << "\tTRIANGLEBUFFER_REFCOUNTING: " << TRIANGLEBUFFER_REFCOUNTING << '\n'
		    << "\tINDEXQUEUEATOMICS: " << INDEXQUEUEATOMICS << '\n'
		    << "\tINDEXQUEUEABORTONOVERFLOW: " << INDEXQUEUEABORTONOVERFLOW << '\n' << '\n'

		    << "\tENFORCE_PRIMITIVE_ORDER: " << ENFORCE_PRIMITIVE_ORDER << '\n'
		    << "\tFORCE_QUAD_SHADING: " << FORCE_QUAD_SHADING << '\n'
		    << "\tBINRASTER_EXCLUSIVE_TILE_ACCESS_MODE: " << BINRASTER_EXCLUSIVE_TILE_ACCESS_MODE << '\n'
		    << "\tTILE_RASTER_EXCLUSIVE_ACCESS_METHOD: " << TILE_RASTER_EXCLUSIVE_ACCESS_METHOD << '\n' << '\n'

		    << "\tVERTEX_FETCH_CS: " << VERTEX_FETCH_CS << '\n' << '\n'

		    << "\tTRACK_FILL_LEVEL: " << TRACK_FILL_LEVEL << '\n'
		    << "\tENABLE_INSTRUMENTATION: " << ENABLE_INSTRUMENTATION << '\n' << '\n'

		    << "\tPATTERN_TECHNIQUE: " << PATTERN_TECHNIQUE << '\n'
		    << "\tOFFSET_PARAMETER: " << OFFSET_PARAMETER << '\n' << '\n';
		return out;
	}

	void initInstrumentation(const cuRE::PipelineModule& module)
	{
		CUfunction init_kernel = module.getFunction("initInstrumentation");

		succeed(cuLaunchKernel(init_kernel, divup(MAX_NUM_BLOCKS, 1024), 1U, 1U, 1024U, 1U, 1U, 0U, 0, nullptr, nullptr));
		succeed(cuCtxSynchronize());
	}
}

namespace cuRE
{
	Instrumentation::Instrumentation(const PipelineModule& module)
		: context(CU::getCurrentContext()),
		  device(CU::getDevice(context)),
		  reset_kernel(module.getFunction("resetInstrumentation")),
		  read_data_kernel(module.getFunction("readInstrumentationData")),
		  mp_index_buffer(module.getGlobal("mp_index")),
		  per_block_timing_buffer(CU::allocMemory(16U * MAX_NUM_BLOCKS * NUM_PER_BLOCK_TIMERS)),
		  triangle_buffer(module.getGlobal("triangle_buffer")),
		  rasterizer_queues(module.getGlobal("rasterizer_queue")),
		  //prof_counter_events(CUPTI::createEventGroup(context)),
		  mp_indices(MAX_NUM_BLOCKS),
		  max_queue_fill_levels(MAX_NUM_RASTERIZERS + 1),
		  num_multiprocessors(CU::getDeviceAttribute<CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT>(CU::getDevice(CU::getCurrentContext())))
	{
		identify(std::cout);

		if (!ENABLE_INSTRUMENTATION)
			return;

		initInstrumentation(module);

		for (auto& t : per_block_timings)
			t.resize(MAX_NUM_BLOCKS);

		//static const char* prof_counters[] = {
		//	"prof_trigger_00",
		//	"prof_trigger_01",
		//	"prof_trigger_02",
		//	//"prof_trigger_03",
		//	//"prof_trigger_04",
		//	//"prof_trigger_05",
		//	//"prof_trigger_06",
		//	//"prof_trigger_07"
		//};

		//for (auto name : prof_counters)
		//{
		//	auto e = CUPTI::getEventID(device, name);
		//	succeed(cuptiEventGroupAddEvent(prof_counter_events, e));
		//}

		//CUPTI::setEventGroupAttribute<CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES>(prof_counter_events, 1);

		//CUpti_EventDomainID domain = CUPTI::getEventGroupAttribute<CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID>(prof_counter_events);
		//uint32_t num_instances = CUPTI::getEventDomainAttribute<CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT>(device, domain);
		//uint32_t num_total_instances = CUPTI::getEventDomainAttribute<CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT>(device, domain);
		//num_prof_counter_events = CUPTI::getEventGroupAttribute<CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS>(prof_counter_events);
		//num_prof_counter_instances = num_instances;

		//succeed(cuptiSetEventCollectionMode(context, CUPTI_EVENT_COLLECTION_MODE_KERNEL));

		//succeed(cuptiEventGroupEnable(prof_counter_events));
		//succeed(cuptiEventGroupResetAllEvents(prof_counter_events));

		reset();
	}

	void Instrumentation::reset()
	{
		if (!ENABLE_INSTRUMENTATION)
			return;

		succeed(cuLaunchKernel(reset_kernel, divup(MAX_NUM_BLOCKS, 1024), 1U, 1U, 1024U, 1U, 1U, 0U, 0, nullptr, nullptr));
		succeed(cuCtxSynchronize());

		//std::memset(&results[0], 0, sizeof(TimingResult) * num_multiprocessors * NUM_PER_BLOCK_TIMERS);
		results = std::make_unique<TimingResult[]>(num_multiprocessors * NUM_PER_BLOCK_TIMERS);
	}

	void Instrumentation::record(unsigned int num_blocks_launched)
	{
		if (TRACK_FILL_LEVEL)
		{
			succeed(cuMemcpyDtoH(&max_queue_fill_levels[0], triangle_buffer + 8, 4U));
			succeed(cuMemcpyDtoH(&max_queue_fill_levels[0] + 1, rasterizer_queues + 4U * 3 * (MAX_NUM_RASTERIZERS + 1023U) / 1024U * 1024U, MAX_NUM_RASTERIZERS * 4U));
		}

		if (!ENABLE_INSTRUMENTATION)
			return;

		//std::vector<std::uint64_t> value_buffer(num_prof_counter_events * num_prof_counter_instances);
		//std::vector<CUpti_EventID> event_ids(num_prof_counter_events);

		//size_t value_buffer_size = value_buffer.size() * sizeof(uint64_t);
		//size_t event_array_size = event_ids.size() * sizeof(CUpti_EventID);
		//size_t num_values_read;
		//succeed(cuCtxSynchronize());
		//succeed(cuptiEventGroupReadAllEvents(prof_counter_events, CUPTI_EVENT_READ_FLAG_NONE, &value_buffer_size, &value_buffer[0], &event_array_size, &event_ids[0], &num_values_read));
		//succeed(cuptiEventGroupResetAllEvents(prof_counter_events));

		CUdeviceptr per_block_timing_buffer_ptr = per_block_timing_buffer;
		unsigned int buffer_stride = 16U * MAX_NUM_BLOCKS;
		void* params[] = {
			&per_block_timing_buffer_ptr,
			&buffer_stride,
			&num_blocks_launched
		};
		succeed(cuLaunchKernel(read_data_kernel, divup(num_blocks_launched, 1024U), 1U, 1U, 1024U, 1U, 1U, 0U, 0, params, nullptr));
		succeed(cuCtxSynchronize());

		succeed(cuMemcpyDtoH(&mp_indices[0], mp_index_buffer, 1U * num_blocks_launched));
		for (int i = 0; i < NUM_PER_BLOCK_TIMERS; ++i)
			succeed(cuMemcpyDtoH(&per_block_timings[i][0], per_block_timing_buffer + i * 16U * MAX_NUM_BLOCKS, 16U * num_blocks_launched));


		std::vector<std::tuple<int, std::uint64_t, int>> tt(num_blocks_launched);

		for (unsigned int i = 0; i < num_blocks_launched; ++i)
		{
			for (int j = 0; j < NUM_PER_BLOCK_TIMERS; ++j)
			{
				std::get<0>(tt[i]) = mp_indices[i];
				std::get<1>(tt[i]) += per_block_timings[j][i].t;
				std::get<2>(tt[i]) = i;
			}
		}

		std::sort(begin(tt), end(tt), [](const auto& a, const auto& b) { return std::get<1>(a) > std::get<1>(b); });
		std::stable_sort(begin(tt), end(tt), [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
		tt.erase(std::unique(begin(tt), end(tt), [](const auto& a, const auto& b) { return std::get<0>(a) == std::get<0>(b); }), end(tt));

		for (int i = 0; i < num_multiprocessors; ++i)
		{
			for (int j = 0; j < NUM_PER_BLOCK_TIMERS; ++j)
			{
				auto mpid = std::get<0>(tt[i]);
				auto blockid = std::get<2>(tt[i]);

				auto id = mpid * NUM_PER_BLOCK_TIMERS + j;
				results[id].N = 1;
				results[id].t = per_block_timings[j][blockid].t;
			}
		}

		//for (unsigned int i = 0; i < num_blocks_launched; ++i)
		//{
		//	for (int j = 0; j < NUM_PER_BLOCK_TIMERS; ++j)
		//	{
		//		auto id = mp_indices[i] * NUM_PER_BLOCK_TIMERS + j;
		//		results[id].N = 1;// += per_block_timings[j][i].N;
		//		results[id].t = std::max(per_block_timings[j][i].t, results[id].t);
		//	}
		//}
	}

	void Instrumentation::reportQueueSizes(PerformanceDataCallback& perf_mon) const
	{
		if (TRACK_FILL_LEVEL)
		{
			std::vector<std::uint32_t> queue_sizes(max_queue_fill_levels.size(), RASTERIZER_QUEUE_SIZE);
			queue_sizes[0] = TRIANGLE_BUFFER_SIZE;
			perf_mon.recordQueueSize(&queue_sizes[0], static_cast<int>(queue_sizes.size()));
		}
	}

	void Instrumentation::reportMaxQueueFillLevel(PerformanceDataCallback& perf_mon) const
	{
		if (TRACK_FILL_LEVEL)
		{
			perf_mon.recordMaxQueueFillLevel(&max_queue_fill_levels[0], static_cast<int>(max_queue_fill_levels.size()));
		}
	}

	void Instrumentation::reportTelemetry(PerformanceDataCallback& perf_mon)
	{
		reportMaxQueueFillLevel(perf_mon);

		if (!ENABLE_INSTRUMENTATION)
			return;

		perf_mon.recordInstrumentationTimers(std::move(results), NUM_PER_BLOCK_TIMERS, num_multiprocessors);

		reset();
	}
}
