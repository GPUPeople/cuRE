


#ifndef INCLUDED_FREEPIPE_INTERMEDIATEGEOMETRYSTORAGE
#define INCLUDED_FREEPIPE_INTERMEDIATEGEOMETRYSTORAGE

#pragma once

#include "config.h"

namespace FreePipe
{

	class IntermediateGeometryStorage
	{
	public:
		float processedVertices[24*MAX_TEMP_VERTICES];
	public:
		unsigned int indices[MAX_TEMP_INDICES];

		unsigned int vertexCounter;
		unsigned int indexCounter;

		__device__ IntermediateGeometryStorage() { }

		__device__ void init()
		{
			vertexCounter = 0;
			indexCounter = 0;
		}
		__device__ unsigned int requestVertices(unsigned int num)
		{
			return atomicAdd(&vertexCounter, num);
		}
		__device__ unsigned int requestIndices(unsigned int num)
		{
			return atomicAdd(&indexCounter, num);
		}

		template<class VertexFormat>
		__device__ VertexFormat* accessProcessedVertices()
		{
			return reinterpret_cast<VertexFormat*>(processedVertices);
		}
	};

}
#endif // INCLUDED_FREEPIPE_INTERMEDIATEGEOMETRYSTORAGE
