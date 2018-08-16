


#ifndef INCLUDED_CURE_INSTRUMENTATION_CONFIG
#define INCLUDED_CURE_INSTRUMENTATION_CONFIG

#pragma once

typedef TimerConfig <
	ENABLE_INSTRUMENTATION,
	true,   // pipeline 0
	true,   // geometry stage 1
	true,   // vertex processing 2
	true,   // triangle processing 3
	true,   // rasterization stage 4
	true,   // bin rasterizer 5
	true,   // tile rasterizer 6
	true,   // bin work assignment 7
	true,   // tile work assignment 8
	false,  // tile marking 9
	false,  // unused
	false,  // progress update 11
	true,   // actual sort 12
	true,   // rop 13
	false,  //num hit bins 14
	false,  //get hit bin 15
	true,   //fragment shader 16
	true    // fragment work assingment 17
> Timers;

#endif  // INCLUDED_CURE_INSTRUMENTATION_CONFIG
