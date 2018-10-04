


#ifndef INCLUDED_CURE_GPU_CONFIGS
#define INCLUDED_CURE_GPU_CONFIGS

#pragma once

#include "cure_config.h"


using GTX_780 =             PipelineConfig<12, 2, 8>;
using GTX_780_Ti =          PipelineConfig<15, 2, 8>;
using GTX_780_Ti_dynamic =  PipelineConfig<15, 2, 8, 8 * 15 * 2>;
using GTX_TITAN =           PipelineConfig<14, 2, 8>;
using MX_150 =              PipelineConfig< 3, 2, 16>;
using MX_150_dynamic =      PipelineConfig< 3, 2, 16, 8 * 3 * 2>;
using GTX_960 =             PipelineConfig< 7, 2, 16>;
using GTX_970 =             PipelineConfig<13, 2, 16>;
using GTX_980 =             PipelineConfig<16, 2, 16>;
using GTX_980_Ti =          PipelineConfig<22, 2, 16>;
using GTX_980_Ti_dynamic =  PipelineConfig<22, 2, 16, 8 * 22 * 2>;
using GTX_TITAN_X =         PipelineConfig<24, 2, 16>;
using GTX_1060 =            PipelineConfig<10, 2, 16>;
using GTX_1060_dynamic =    PipelineConfig<10, 2, 16, 8 * 10 * 2>;
using GTX_1070 =            PipelineConfig<15, 2, 16>;
using GTX_1070_Ti =         PipelineConfig<19, 2, 16>;
using GTX_1080 =            PipelineConfig<20, 2, 16>;
using GTX_1080_dynamic =    PipelineConfig<20, 2, 16, 8 * 20 * 2>;
using GTX_1080_Ti =         PipelineConfig<28, 2, 16>;
using TITAN_X =             GTX_1080_Ti;
using GTX_1080_Ti_dynamic = PipelineConfig<28, 2, 16, 8 * 28 * 2>;
using TITAN_X_dynamic =     GTX_1080_Ti_dynamic;
using TITAN_Xp =            PipelineConfig<30, 2, 16>;


constexpr unsigned int WARP_SIZE = 32U;

#endif  // INCLUDED_CURE_GPU_CONFIGS
