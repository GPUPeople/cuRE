


#ifndef INCLUDED_CURE_PIPELINE_CONFIG_SPACE
#define INCLUDED_CURE_PIPELINE_CONFIG_SPACE

#pragma once

#include <meta_utils.h>


enum class TILE_ACCESS_MODE { WARP_EXCLUSIVE = 0, WARP_PER_FRAGMENT = 1, COVERAGE_MASK = 2 }; //0 = exclusive tiles, 1 = warp per tile and fragment based resolution, 2 = coverage mask tile raster
enum class FRAMEBUFFER_SYNC_METHOD { NO_SYNC = 0, SYNC_ALL = 1, SYNC_FRAGMENTS = 2, SUBTILE_SYNC = 3, MASK_SYNC = 4, POLLED_MASK_SYNC = 5}; //0 = none, 1-2/5 different versions

enum class PATTERNTECHNIQUE { DIAGONAL, OFFSET, OFFSET_SHIFT, OFFSET_SHIFT_SLIM, DIAGONAL_ITERATIVE, OFFSET_ITERATIVE, OFFSET_SHIFT_ITERATIVE };

template <unsigned int NUM_MULTIPROCESSORS, unsigned int BLOCKS_PER_MULTIPROCESSOR, unsigned int WARPS_PER_BLOCK, unsigned int DYNAMIC_RASTERIZERS = 0>
struct PipelineConfig {};

template <class... Configs>
struct PipelineConfigList;

template <unsigned int NUM_MULTIPROCESSORS, unsigned int BLOCKS_PER_MULTIPROCESSOR, unsigned int WARPS_PER_BLOCK, unsigned int DYNAMIC_RASTERIZERS>
struct PipelineConfigList<PipelineConfig<NUM_MULTIPROCESSORS, BLOCKS_PER_MULTIPROCESSOR, WARPS_PER_BLOCK, DYNAMIC_RASTERIZERS>>
{
	static constexpr int MAX_NUM_BLOCKS = NUM_MULTIPROCESSORS * BLOCKS_PER_MULTIPROCESSOR;
	static constexpr int MAX_WARPS_PER_BLOCK = WARPS_PER_BLOCK;
	static constexpr int MAX_NUM_RASTERIZERS = static_max<MAX_NUM_BLOCKS, DYNAMIC_RASTERIZERS>::value;
};

template <unsigned int... NUM_MULTIPROCESSORS, unsigned int... BLOCKS_PER_MULTIPROCESSOR, unsigned int... WARPS_PER_BLOCK, unsigned int... DYNAMIC_RASTERIZERS>
struct PipelineConfigList<PipelineConfig<NUM_MULTIPROCESSORS, BLOCKS_PER_MULTIPROCESSOR, WARPS_PER_BLOCK, DYNAMIC_RASTERIZERS>...>
{
	static constexpr int MAX_NUM_BLOCKS = static_max<NUM_MULTIPROCESSORS * BLOCKS_PER_MULTIPROCESSOR...>::value;
	static constexpr int MAX_WARPS_PER_BLOCK = static_max<WARPS_PER_BLOCK...>::value;
	static constexpr int MAX_NUM_RASTERIZERS = static_max<MAX_NUM_BLOCKS, DYNAMIC_RASTERIZERS...>::value;
};


template <bool ENABLE_INSTRUMENTATION, bool... enable>
struct TimerConfig
{
	static constexpr unsigned int NUM_TIMERS = sizeof...(enable);

	template <unsigned int i, bool... Tail>
	struct IsEnabled;

	template <bool Head, bool... Tail>
	struct IsEnabled<0, Head, Tail...>
	{
		static constexpr bool value = Head && ENABLE_INSTRUMENTATION;
	};

	template <unsigned int i, bool Head, bool... Tail>
	struct IsEnabled<i, Head, Tail...>
	{
		static constexpr bool value = IsEnabled<i - 1, Tail...>::value;
	};

	template <unsigned int i>
	using Enabled = IsEnabled<i, enable...>;
};

#endif  // INCLUDED_CURE_PIPELINE_CONFIG_SPACE
