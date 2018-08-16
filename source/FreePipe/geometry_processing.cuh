//
//
//
//#ifndef INCLUDED_FREEPIPE_GEOMETRY_PROCESSING
//#define INCLUDED_FREEPIPE_GEOMETRY_PROCESSING
//
//#pragma once
//
//#include "config.h"
//
//#include "shaders/vertex_simple.cuh"
//#include "IntermediateGeometryStorage.cuh"
//
//#include "tools/bitonicSort.cuh"
//#include "tools/prefixSum.cuh"
//
//#include "ptx_primitives.cuh"
//
//
//namespace FreePipe
//{
//	__device__ IntermediateGeometryStorage vStorageSimpleVertex;
//}
//
//extern "C"
//{
//	__constant__ float *c_positions, *c_normals, *c_texCoords;
//	__constant__ unsigned int *c_indices, *c_patchData;
//
//	__global__ void initVstorageSimpleVertex()
//	{
//		using namespace FreePipe;
//		vStorageSimpleVertex.init();
//	}
//}
//
//namespace FreePipe
//{
//	template<class VertexShader>
//	__device__ void process_geometry_vertices(float* positions, float* normals, float* texCoords, unsigned int* indices, unsigned int *patchData, IntermediateGeometryStorage& geometryOutStorage, unsigned int num_vertices, unsigned int num_indices)
//	{
//		unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
//
//		if (tid == 0)
//		{
//			// set output buffer counts
//			geometryOutStorage.requestVertices(num_vertices);
//			geometryOutStorage.requestIndices(num_indices);
//		}
//
//		// process vertices
//		for (unsigned int vId = tid; vId < num_vertices; vId += blockDim.x*gridDim.x)
//		{
//#if CLIPSPACE_GEOMETRY
//            math::float4 pos = math::float4(positions[vId * 4], positions[vId * 4 + 1], positions[vId * 4 + 2], positions[vId * 4 + 3]);
//            typename VertexShader::VertexOut out = VertexShader::process(pos);
//#else
//			math::float3 pos = math::float3(positions[vId * 3], positions[vId * 3 + 1], positions[vId * 3 + 2]);
//			math::float3 normal = math::float3(normals[vId * 3], normals[vId * 3 + 1], normals[vId * 3 + 2]);
//			math::float2 tex = math::float2(texCoords[vId * 2], texCoords[vId * 2 + 1]);
//            typename VertexShader::VertexOut out = VertexShader::process(pos, normal, tex);
//#endif
//
//			geometryOutStorage.accessProcessedVertices<typename VertexShader::VertexOut>()[vId] = out;
//		}
//
//		// copy indices
//		for (unsigned int iId = tid; iId < num_indices; iId += blockDim.x*gridDim.x)
//		{
//			geometryOutStorage.indices[iId] = indices[iId];
//		}
//	};
//
//	template<class VertexShader>
//	__device__ void process_geometry_simple(float* positions, float* normals, float* texCoords, unsigned int* indices, unsigned int *patchData, IntermediateGeometryStorage& geometryOutStorage, int patchId)
//	{
//		// blocksize must be a multiple of 3!
//
//		// get my patchdata
//		int myPatch = patchId;
//		unsigned int startIndex = patchData[myPatch];
//		unsigned int endIndex = patchData[myPatch + 1];
//
//
//		// for now, just create loads of duplicates
//		__shared__ int vertexWriteoutOffset, indexWriteoutOffset;
//		for (int nullId = startIndex; nullId < endIndex; nullId += blockDim.x)
//		{
//			// get offsets
//			if (threadIdx.x == 0)
//			{
//				unsigned int num = min(endIndex - nullId, blockDim.x);
//				vertexWriteoutOffset = geometryOutStorage.requestVertices(num);
//				indexWriteoutOffset = geometryOutStorage.requestIndices(num);
//			}
//			__syncthreads();
//
//			int myId = nullId + threadIdx.x;
//			if (myId < endIndex)
//			{ 
//				unsigned int vId = indices[myId];
//				unsigned int myVertexOutId = vertexWriteoutOffset + threadIdx.x;
//				unsigned int myIndexOutId = indexWriteoutOffset + threadIdx.x;
//
//#if CLIPSPACE_GEOMETRY
//                math::float4 pos = math::float4(positions[vId * 4], positions[vId * 4 + 1], positions[vId * 4 + 2], positions[vId * 4 + 3]);
//                typename VertexShader::VertexOut out = VertexShader::process(pos);
//#else
//				math::float3 pos = math::float3(positions[vId * 3], positions[vId * 3 + 1], positions[vId * 3 + 2]);
//				math::float3 normal = math::float3(normals[vId * 3], normals[vId * 3 + 1], normals[vId * 3 + 2]);
//				math::float2 tex = math::float2(texCoords[vId * 2], texCoords[vId * 2 + 1]);
//				typename VertexShader::VertexOut out = VertexShader::process(pos, normal, tex);
//#endif
//
//				// write results to geometryOutStorage
//				geometryOutStorage.accessProcessedVertices<typename VertexShader::VertexOut>()[myVertexOutId] = out;
//				geometryOutStorage.indices[myIndexOutId] = myVertexOutId;
//			}
//		}
//	};
//
//
//	template<unsigned int TableSize> 
//	__device__ unsigned int hash(unsigned int inId, unsigned int offset = 0)
//	{
//		//return (inId + offset) % TableSize;
//		return (inId * 3407 + offset) % TableSize;
//	}
//
//	template<class VertexShader, int BlockSize, bool Collaborative>
//	__device__ void process_geometry_hashing(float* positions, float* normals, float* texCoords, unsigned int* indices, unsigned int *patchData, IntermediateGeometryStorage& geometryOutStorage, int patchId)
//	{
//		// search tries alone
//		const int SeqSoloSearch = 2;
//
//		// get my patchdata
//		int myPatch = patchId;
//		unsigned int startIndex = patchData[myPatch];
//		unsigned int endIndex = patchData[myPatch + 1];
//
//		extern __shared__ unsigned int s_data[];
//
//		// 1. clear hash table
//		unsigned int* hashtable = s_data;
//		hashtable[threadIdx.x] = 0xFFFFFFFF;
//
//		// 2. alloc vertices and indices
//		__shared__ int vertexWriteoutOffset, indexWriteoutOffset;
//		if (threadIdx.x == 0)
//		{
//			vertexWriteoutOffset = geometryOutStorage.requestVertices(GPM_PATCH_MAX_VERTICES);
//			indexWriteoutOffset = geometryOutStorage.requestIndices(endIndex - startIndex);
//		}
//		__syncthreads();
//
//		// 3. load indices and insert into hash table - assign individual spots + generate adjusted index buffer
//		#pragma unroll
//		for (int readId = startIndex + threadIdx.x; readId < endIndex; readId += BlockSize)
//		{
//			unsigned int myId = indices[readId];
//			unsigned int p = hash<BlockSize>(myId);
//			for (unsigned int i = 0; i < SeqSoloSearch || Collaborative; ++i)
//			{
//				unsigned int prev = atomicCAS(hashtable + p, 0xFFFFFFFF, myId);
//				// run sequential search at first
//				if (prev == 0xFFFFFFFF || prev == myId)
//				{
//					geometryOutStorage.indices[indexWriteoutOffset + readId - startIndex] = vertexWriteoutOffset + p;
//					p = 0xFFFFFFFF;
//					break;
//				}
//				p = (p + 1) % BlockSize;
//			}
//
//			if (Collaborative)
//			{
//				// optimized warp search
//				unsigned int notdonemask = __ballot_sync(~0U, p != 0xFFFFFFFF);
//				while (notdonemask != 0u)
//				{
//					unsigned int possiblemask;
//					do
//					{
//						// get positions and id from the next not done thread
//						unsigned int active = __ffs(notdonemask) - 1;
//						unsigned int otherId = __shfl_sync(~0U, myId, active);
//						unsigned int otherP = __shfl_sync(~0U, p, active);
//						otherP = (otherP + laneid()) % BlockSize;
//						unsigned int val = hashtable[otherP];
//						unsigned int tpossiblemask = __ballot_sync(~0U, val == otherId || val == 0xFFFFFFFF);
//						// store possibility mask
//						if (laneid() == active)
//							possiblemask = tpossiblemask;
//						notdonemask = notdonemask & (~(1u << active));
//					} while (notdonemask != 0u);
//
//					// run through possibilites
//					unsigned int offset = __ffs(possiblemask);
//					while (offset != 0)
//					{
//						unsigned int tp = (p + offset - 1) % BlockSize;
//						unsigned int prev = atomicCAS(hashtable + tp, 0xFFFFFFFF, myId);
//						if (prev == 0xFFFFFFFF || prev == myId)
//						{
//							geometryOutStorage.indices[indexWriteoutOffset + readId - startIndex] = vertexWriteoutOffset + tp;
//							p = 0xFFFFFFFF;
//							break;
//						}
//						offset = __ffs(possiblemask & (~((1u << offset) - 1)));
//					}
//					if (p != 0xFFFFFFFF)
//						p = p + 32u;
//					notdonemask = __ballot_sync(~0U, p != 0xFFFFFFFF);
//				}
//			}
//		}
//
//		__syncthreads();
//
//		// 4. process vertex (every thread is assigned a maximum of 1 vertex)
//		unsigned int vId = hashtable[threadIdx.x];
//		if (vId != 0xFFFFFFFF)
//		{
//#if CLIPSPACE_GEOMETRY
//            math::float4 pos = math::float4(positions[vId * 4], positions[vId * 4 + 1], positions[vId * 4 + 2], positions[vId * 4 + 3]);
//            typename VertexShader::VertexOut out = VertexShader::process(pos);
//#else
//			math::float3 pos = math::float3(positions[vId * 3], positions[vId * 3 + 1], positions[vId * 3 + 2]);
//			math::float3 normal = math::float3(normals[vId * 3], normals[vId * 3 + 1], normals[vId * 3 + 2]);
//			math::float2 tex = math::float2(texCoords[vId * 2], texCoords[vId * 2 + 1]);
//			typename VertexShader::VertexOut out = VertexShader::process(pos, normal, tex);
//#endif
//
//			int myVertexOutId = vertexWriteoutOffset + threadIdx.x;
//			geometryOutStorage.accessProcessedVertices<typename VertexShader::VertexOut>()[myVertexOutId] = out;
//		}
//	};
//
//	template<class VertexShader, int BlockSize>
//	__device__ void process_geometry_sorting(float* positions, float* normals, float* texCoords, unsigned int* indices, unsigned int *patchData, IntermediateGeometryStorage& geometryOutStorage, int patchId)
//	{
//		// get my patchdata
//		int myPatch = patchId;
//		unsigned int startIndex = patchData[myPatch];
//		unsigned int endIndex = patchData[myPatch + 1];
//
//		extern __shared__ unsigned int s_data[];
//
//		// 1. load ids to shared memory and remove duplicated vertices from vertex processing 
//		unsigned int* patchIds = s_data;
//		unsigned int* processPatchIds = patchIds + GPM_PATCH_MAX_INDICES + 1;
//
//#pragma unroll
//		for (int offsetid = 0; offsetid < GPM_PATCH_MAX_INDICES + 1; offsetid += BlockSize)
//		{
//			int readId = startIndex + offsetid + threadIdx.x;
//			unsigned int myId = 0xFFFFFFFF;
//			if (readId < endIndex)
//				myId = indices[readId];
//			patchIds[offsetid + threadIdx.x] = myId;
//			processPatchIds[offsetid + threadIdx.x] = offsetid + threadIdx.x;
//		}
//
//		// 2. sort ids
//		Sort::bitonic<unsigned int, unsigned int, GPM_PATCH_MAX_INDICES + 1, BlockSize, true >(patchIds, processPatchIds, threadIdx.x);
//
//		//keep processPatchIds in registers
//		const int PerThreadVars = (GPM_PATCH_MAX_INDICES + BlockSize) / BlockSize;
//		int idOrigin[PerThreadVars];
//		for (int i = 0; i < PerThreadVars; ++i)
//			idOrigin[i] = processPatchIds[i * BlockSize + threadIdx.x];
//
//		__syncthreads();
//
//		// 3. generate local offsets
//#pragma unroll
//		for (int offsetid = 0; offsetid < GPM_PATCH_MAX_INDICES + 1; offsetid += BlockSize)
//		{
//			int val = 0;
//			if (offsetid + threadIdx.x < GPM_PATCH_MAX_INDICES)
//				val = (patchIds[offsetid + threadIdx.x] != patchIds[offsetid + threadIdx.x + 1]);
//			processPatchIds[offsetid + threadIdx.x] = val;
//		}
//
//		// numVertices is only valid for first thread!
//		int numVertices0 = Prefix::exclusive<unsigned int, GPM_PATCH_MAX_INDICES + 1, BlockSize>(processPatchIds, threadIdx.x);
//
//		// alloc vertices 
//		__shared__ int vertexWriteoutOffset, indexWriteoutOffset, numVertices;
//		if (threadIdx.x == 0)
//		{
//			numVertices = numVertices0;
//			vertexWriteoutOffset = geometryOutStorage.requestVertices(numVertices);
//			indexWriteoutOffset = geometryOutStorage.requestIndices(endIndex - startIndex);
//		}
//
//
//		// load results to registers 
//		int offsets[PerThreadVars];
//#pragma unroll
//		for (int i = 0; i < PerThreadVars; ++i)
//			offsets[i] = processPatchIds[i * BlockSize + threadIdx.x];
//		__syncthreads();
//
//		// set where every id will be found + clear processPatchIds array
//#pragma unroll
//		for (int i = 0; i < PerThreadVars; ++i)
//			processPatchIds[idOrigin[i]] = offsets[i];
//
//
//		//load first patchIdsToRegister / only the first will be changed!
//		unsigned int patchIdFirst = patchIds[threadIdx.x];
//
//		__syncthreads();
//
//		// set who will be processing which id
//
//		patchIds[offsets[0]] = patchIdFirst;
//
//#pragma unroll
//		for (int i = 1; i < PerThreadVars; ++i)
//			patchIds[offsets[i]] = patchIds[i * BlockSize + threadIdx.x];
//
//		__syncthreads();
//
//		// 4. process vertex (every thread is assigned a maximum of 1 vertex)
//		unsigned int vId = patchIds[threadIdx.x];
//		if (threadIdx.x < numVertices)
//		{
//#if CLIPSPACE_GEOMETRY
//            math::float4 pos = math::float4(positions[vId * 4], positions[vId * 4 + 1], positions[vId * 4 + 2], positions[vId * 4 + 3]);
//            typename VertexShader::VertexOut out = VertexShader::process(pos);
//#else
//			math::float3 pos = math::float3(positions[vId * 3], positions[vId * 3 + 1], positions[vId * 3 + 2]);
//			math::float3 normal = math::float3(normals[vId * 3], normals[vId * 3 + 1], normals[vId * 3 + 2]);
//			math::float2 tex = math::float2(texCoords[vId * 2], texCoords[vId * 2 + 1]);
//			typename VertexShader::VertexOut out = VertexShader::process(pos, normal, tex);
//#endif
//
//			int myVertexOutId = vertexWriteoutOffset + threadIdx.x;
//			geometryOutStorage.accessProcessedVertices<typename VertexShader::VertexOut>()[myVertexOutId] = out;
//		}
//
//		// 5. generate adjusted index buffer
//#pragma unroll
//		for (int offsetid = 0; offsetid < GPM_PATCH_MAX_INDICES + 1; offsetid += BlockSize)
//		{
//			if (offsetid + threadIdx.x < endIndex - startIndex)
//				geometryOutStorage.indices[indexWriteoutOffset + offsetid + threadIdx.x] = vertexWriteoutOffset + processPatchIds[offsetid + threadIdx.x];
//		}
//	};
//
//	template<class VertexShader>
//	__device__ void process_geometry_warp(float* positions, float* normals, float* texCoords, unsigned int* indices, unsigned int *patchData, IntermediateGeometryStorage& geometryOutStorage, int patchId)
//	{
//		// get my patchdata
//		int myPatch = patchId;
//		unsigned int startIndex = patchData[myPatch];
//		unsigned int endIndex = patchData[myPatch + 1];
//		unsigned int lId = laneid();
//
//		// reserve output buffers
//		unsigned int indexWriteoutOffset, vertexWriteoutOffset;
//		if (lId == 0)
//			vertexWriteoutOffset = geometryOutStorage.requestVertices(32),
//			indexWriteoutOffset = geometryOutStorage.requestIndices(endIndex - startIndex);
//
//		indexWriteoutOffset = __shfl_sync(~0U, indexWriteoutOffset, 0) - startIndex;
//		vertexWriteoutOffset = __shfl_sync(~0U, vertexWriteoutOffset, 0);
//
//		// warp wide merging
//		unsigned int fillCounter = 0u;
//		unsigned int myId = 0xFFFFFFFF;
//
//
//
//		for (unsigned int readOffset = startIndex; readOffset < endIndex; readOffset += 32u)
//		{
//			unsigned int incomingId = 0xFFFFFFFF;
//			unsigned int outGoingId = 0xFFFFFFFF;
//			if (readOffset + lId < endIndex)
//				incomingId = indices[readOffset + lId];
//
//			// voting
//			#pragma unroll
//			for (unsigned int i = 0; i < 32u; ++i)
//			{
//				unsigned int current = __shfl_sync(~0U, incomingId, i);
//				unsigned int matchMask = __ballot_sync(~0U, current == myId);
//
//				if (matchMask == 0)
//				{
//					if (fillCounter == lId)
//						myId = current;
//					matchMask = 1u << fillCounter;
//					++fillCounter;
//				}
//				
//
//				if (i == lId)
//					outGoingId = matchMask;
//				//outGoingId = outGoingId - (i == lId)*(0xFFFFFFFF - __ffs(matchMask) + 1u);
//				//outGoingId = outGoingId - (i == lId)*(0xFFFFFFFF - matchMask);
//			}
//
//			// write indices out
//			if (incomingId != 0xFFFFFFFF)
//				geometryOutStorage.indices[indexWriteoutOffset + readOffset + lId] = vertexWriteoutOffset + __ffs(outGoingId) - 1u;
//		}
//
//		// run vertex shader and write out
//		if (myId != 0xFFFFFFFF)
//		{
//			math::float3 pos = math::float3(positions[myId * 3], positions[myId * 3 + 1], positions[myId * 3 + 2]);
//			math::float3 normal = math::float3(normals[myId * 3], normals[myId * 3 + 1], normals[myId * 3 + 2]);
//			math::float2 tex = math::float2(texCoords[myId * 2], texCoords[myId * 2 + 1]);
//			typename VertexShader::VertexOut out = VertexShader::process(pos, normal, tex);
//
//			int myVertexOutId = vertexWriteoutOffset + lId;
//			geometryOutStorage.accessProcessedVertices<typename VertexShader::VertexOut>()[myVertexOutId] = out;
//		}
//	}
//
//
//	template<class VertexShader>
//	__device__ void process_geometry_warp_nopatching(float* positions, float* normals, float* texCoords, unsigned int* indices, unsigned int *patchData, IntermediateGeometryStorage& geometryOutStorage, int patchId, unsigned int num_indices)
//	{
//		// get my patchdata
//		int myPatch = patchId;
//		unsigned int startIndex = myPatch*GPM_WARP_NOPREPATCHING_INDICES;
//		unsigned int endIndex = min(num_indices,(myPatch + 1)*GPM_WARP_NOPREPATCHING_INDICES);
//		unsigned int lId = laneid();
//
//		extern __shared__ unsigned int s_data[];
//		unsigned int* per_warp_storage = s_data + threadIdx.x / 32 * GPM_WARP_NOPREPATCHING_INDICES;
//
//		
//		while (startIndex < endIndex)
//		{
//			
//			// warp wide merging
//			unsigned int fillCounter = 0u;
//			unsigned int myId = 0xFFFFFFFF;
//			unsigned int processedIndices = 0u;
//
//			unsigned int readOffset = startIndex;
//			for (; readOffset < endIndex && fillCounter < 32u; readOffset += 32u)
//			{
//				// read incoming 
//				unsigned int incomingId = 0xFFFFFFFF;
//				unsigned int outGoingId = 0xFFFFFFFF;
//				if (readOffset + lId < endIndex)
//					incomingId = indices[readOffset + lId];
//
//				// voting
//				#pragma unroll
//				for (unsigned int i = 0; i < 32u; ++i)
//				{
//					unsigned int current = __shfl_sync(~0U, incomingId, i);
//					unsigned int matchMask = __ballot_sync(~0U, current == myId);
//
//					if (matchMask == 0)
//					{
//						if (fillCounter == lId)
//							myId = current;
//						matchMask = 1u << fillCounter;
//						++fillCounter;
//					}
//
//					if (i == lId)
//						outGoingId = matchMask;
//				}
//
//				// write indices out
//				per_warp_storage[processedIndices + lId] = __ffs(outGoingId) - 1u;
//				processedIndices += min(32u, __ffs(__ballot_sync(~0U, outGoingId == 0u || incomingId == 0xFFFFFFFF)) - 1u);
//			}
//
//			// the fillCounter determines the number of vertices
//			unsigned int processedVertices = min(fillCounter, 32u);
//
//			// make sure we just process entire triangles
//			processedIndices = processedIndices / 3 * 3;
//
//			// set new startindex based on the number of indices we really could process
//			startIndex = startIndex + processedIndices;
//
//			// reserve output buffers
//			unsigned int indexWriteoutOffset, vertexWriteoutOffset;
//			if (lId == 0)
//				vertexWriteoutOffset = geometryOutStorage.requestVertices(processedVertices),
//				indexWriteoutOffset = geometryOutStorage.requestIndices(processedIndices);
//
//			indexWriteoutOffset = __shfl_sync(~0U, indexWriteoutOffset, 0);
//			vertexWriteoutOffset = __shfl_sync(~0U, vertexWriteoutOffset, 0);
//
//			// write indices out
//			for (int i = lId; i < processedIndices; i += 32)
//				geometryOutStorage.indices[indexWriteoutOffset + i] = vertexWriteoutOffset + per_warp_storage[i];
//
//			// run vertex shader and write out
//			if (lId < processedVertices)
//			{
//#if CLIPSPACE_GEOMETRY
//                math::float4 pos = math::float4(positions[myId * 4], positions[myId * 4 + 1], positions[myId * 4 + 2], positions[myId * 4 + 3]);
//                typename VertexShader::VertexOut out = VertexShader::process(pos);
//#else
//				math::float3 pos = math::float3(positions[myId * 3], positions[myId * 3 + 1], positions[myId * 3 + 2]);
//				math::float3 normal = math::float3(normals[myId * 3], normals[myId * 3 + 1], normals[myId * 3 + 2]);
//				math::float2 tex = math::float2(texCoords[myId * 2], texCoords[myId * 2 + 1]);
//				typename VertexShader::VertexOut out = VertexShader::process(pos, normal, tex);
//#endif
//
//				int myVertexOutId = vertexWriteoutOffset + lId;
//				geometryOutStorage.accessProcessedVertices<typename VertexShader::VertexOut>()[myVertexOutId] = out;
//			}
//		}
//	}
//	
//	
//	template<unsigned int Processing, class VertexShader>
//	struct GeometrySeparateStage;
//
//
//	template<class VertexShader>
//	struct GeometrySeparateStage<GPM_ALLVERTICES, VertexShader>
//	{
//		static __device__ void run(unsigned int patches, unsigned int num_vertices, unsigned int num_indices)
//		{
//			process_geometry_vertices<VertexShader>(c_positions, c_normals, c_texCoords, c_indices, c_patchData, FreePipe::vStorageSimpleVertex, num_vertices, num_indices);
//		}
//	};
//
//	template<class VertexShader>
//	struct GeometrySeparateStage<GPM_ALLINDICES, VertexShader>
//	{
//		static __device__ void run(unsigned int patches, unsigned int num_vertices, unsigned int num_indices)
//		{
//			process_geometry_simple<VertexShader>(c_positions, c_normals, c_texCoords, c_indices, c_patchData, FreePipe::vStorageSimpleVertex, blockIdx.x);
//		}
//	};
//
//	template<class VertexShader>
//	struct GeometrySeparateStage<GPM_SORTING, VertexShader>
//	{
//		static __device__ void run(unsigned int patches, unsigned int num_vertices, unsigned int num_indices)
//		{
//			process_geometry_sorting<VertexShader, GPM_PATCH_MAX_VERTICES>(c_positions, c_normals, c_texCoords, c_indices, c_patchData, FreePipe::vStorageSimpleVertex, blockIdx.x);
//		}
//	};
//
//	template<class VertexShader>
//	struct GeometrySeparateStage<GPM_HASHING, VertexShader>
//	{
//		static __device__ void run(unsigned int patches, unsigned int num_vertices, unsigned int num_indices)
//		{
//			process_geometry_hashing<VertexShader, GPM_PATCH_MAX_VERTICES, false>(c_positions, c_normals, c_texCoords, c_indices, c_patchData, FreePipe::vStorageSimpleVertex, blockIdx.x);
//		}
//	};
//
//	template<class VertexShader>
//	struct GeometrySeparateStage<GPM_HASHING_COLLABORATIVE, VertexShader>
//	{
//		static __device__ void run(unsigned int patches, unsigned int num_vertices, unsigned int num_indices)
//		{
//			process_geometry_hashing<VertexShader, GPM_PATCH_MAX_VERTICES, true>(c_positions, c_normals, c_texCoords, c_indices, c_patchData, FreePipe::vStorageSimpleVertex, blockIdx.x);
//		}
//	};
//
//	template<class VertexShader>
//	struct GeometrySeparateStage<GPM_WARP_VOTING, VertexShader>
//	{
//		static __device__ void run(unsigned int patches, unsigned int num_vertices, unsigned int num_indices)
//		{
//			unsigned int patchId = blockIdx.x * (KERNEL_THREADS / 32) + threadIdx.x / 32;
//			if (patchId < patches)
//				process_geometry_warp<VertexShader>(c_positions, c_normals, c_texCoords, c_indices, c_patchData, FreePipe::vStorageSimpleVertex, patchId);
//		}
//	};
//
//	template<class VertexShader>
//	struct GeometrySeparateStage<GPM_WARP_VOTING_NOPREPATCHING, VertexShader>
//	{
//		static __device__ void run(unsigned int patches, unsigned int num_vertices, unsigned int num_indices)
//		{
//			unsigned int patchId = blockIdx.x * (KERNEL_THREADS / 32) + threadIdx.x / 32;
//			if (patchId < (num_indices + GPM_WARP_NOPREPATCHING_INDICES - 1) / GPM_WARP_NOPREPATCHING_INDICES)
//				process_geometry_warp_nopatching<VertexShader>(c_positions, c_normals, c_texCoords, c_indices, c_patchData, FreePipe::vStorageSimpleVertex, patchId, num_indices);
//		}
//	};
//	
//}
//
//extern "C"
//{
//
//#if CLIPSPACE_GEOMETRY
//
//    __global__ void runGeometryStageClipSpace(unsigned int patches, unsigned int num_vertices, unsigned int num_indices)
//    {
//        using namespace FreePipe;
//        GeometrySeparateStage<GEOMETRY_PROCESSING, Shaders::ClipSpaceVertexShader>::run(patches, num_vertices, num_indices);
//    }
//
//#else
//
//	__global__ void runGeometryStageSimpleVertex(unsigned int patches, unsigned int num_vertices, unsigned int num_indices)
//	{
//		using namespace FreePipe;
//		GeometrySeparateStage<GEOMETRY_PROCESSING, Shaders::SimpleVertexShader>::run(patches, num_vertices, num_indices);
//	}
//
//	__global__ void runGeometryStageSimpleVertexTex(unsigned int patches, unsigned int num_vertices, unsigned int num_indices)
//	{
//		using namespace FreePipe;
//		GeometrySeparateStage<GEOMETRY_PROCESSING, Shaders::SimpleVertexShaderTex>::run(patches, num_vertices, num_indices);
//	}
//
//	__global__ void runGeometryStageSimpleVertexLight(unsigned int patches, unsigned int num_vertices, unsigned int num_indices)
//	{
//		using namespace FreePipe;
//		GeometrySeparateStage<GEOMETRY_PROCESSING, Shaders::SimpleVertexShaderLight>::run(patches, num_vertices, num_indices);
//	}
//
//	__global__ void runGeometryStageSimpleVertexLightTex(unsigned int patches, unsigned int num_vertices, unsigned int num_indices)
//	{
//		using namespace FreePipe;
//		GeometrySeparateStage<GEOMETRY_PROCESSING, Shaders::SimpleVertexShaderLightTex>::run(patches, num_vertices, num_indices);
//	}
//	
//#endif
//	
//}
//
//
//#endif //INCLUDED_FREEPIPE_FRAGMENT_PROCESSING
