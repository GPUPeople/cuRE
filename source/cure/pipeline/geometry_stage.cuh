


#ifndef INCLUDED_CURE_GEOMETRY_STAGE
#define INCLUDED_CURE_GEOMETRY_STAGE

#pragma once

#ifndef GEOMETRY_STAGE_GLOBAL
#define GEOMETRY_STAGE_GLOBAL extern
#endif

extern "C"
{
	GEOMETRY_STAGE_GLOBAL __constant__ const float4* vertex_buffer;

	GEOMETRY_STAGE_GLOBAL __constant__ const unsigned int* index_buffer;
	GEOMETRY_STAGE_GLOBAL __constant__ unsigned int num_indices;

	GEOMETRY_STAGE_GLOBAL __device__ unsigned int index_counter;
}

#endif  // INCLUDED_CURE_GEOMETRY_STAGE
