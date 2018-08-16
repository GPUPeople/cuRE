


#include <algorithm>
#include <functional>
#include <set>
#include <stdexcept>
#include <vector>

#include <CUDA/error.h>
#include <CUDA/memory.h>
#include <math/vector.h>

#include "config.h"

#include "Geometry.h"
#include "Renderer.h"


namespace FreePipe
{
	IndexedGeometry::IndexedGeometry(Renderer& renderer, CUmodule module, const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices)
	    : vertexBuffer(CU::allocMemory(num_vertices * 12u)),
	      normalBuffer(CU::allocMemory(num_vertices * 12u)),
	      indexBuffer(CU::allocMemory(num_indices * 4u)),
	      num_vertices(num_vertices),
	      num_indices(num_indices),
	      renderer(renderer),
	      module(module)
	{
		// generate buffers and copy data to the GPU
		succeed(cuMemcpyHtoD(vertexBuffer, position, num_vertices * 12u));
		succeed(cuMemcpyHtoD(normalBuffer, normals, num_vertices * 12u));
		succeed(cuMemcpyHtoD(indexBuffer, indices, num_indices * 4u));

		// if texcoord buffer is zero, there are no tex coords
		if (texcoord)
		{
			texcoordBuffer = CU::allocMemory(num_vertices * 8u);
			succeed(cuMemcpyHtoD(texcoordBuffer, texcoord, num_vertices * 8u));
		}

		// generate patches
		std::vector<std::uint32_t> patchStarts;
		unsigned int max_patch_vertices = GPM_PATCH_MAX_VERTICES;
		if (GEOMETRY_PROCESSING == GPM_WARP_VOTING)
			max_patch_vertices = 32;
		patchStarts.reserve(2 * std::max(num_vertices / max_patch_vertices, num_indices / max_patch_vertices));
		std::set<std::uint32_t> seenIndices;

		patchStarts.push_back(0u);
		std::uint32_t currentPatchIndices = 0u;
		std::uint32_t i;
		for (i = 0u; i + 3u <= num_indices; i += 3u, currentPatchIndices += 3u)
		{
			bool restartPatch = false;

			// if the current triangle goes beyond the max index size -> finish patch
			if (currentPatchIndices + 3 > max_patch_vertices)
				restartPatch = true;
			else
			{
				// can we add the triangle while staying within the limit?
				std::uint32_t toAdd = 0;
				std::for_each(indices + i, indices + i + 3, [&](std::uint32_t id) {
					toAdd += seenIndices.find(id) == seenIndices.end() ? 1 : 0;
				});
				if (seenIndices.size() + toAdd >= max_patch_vertices)
					restartPatch = true;
			}
			if (restartPatch)
			{
				patchStarts.push_back(i);
				currentPatchIndices = 0;
				seenIndices.clear();
			}
			std::for_each(indices + i, indices + i + 3, [&](std::uint32_t id) { seenIndices.insert(id); });
		}
		patchStarts.push_back(i);

		// allocate patch buffer and copy patch data
		patchBuffer = CU::allocMemory(patchStarts.size() * 4u);
		succeed(cuMemcpyHtoD(patchBuffer, &patchStarts[0], patchStarts.size() * 4u));
		num_patches = patchStarts.size() - 1;

		// get constants
		size_t constant_size;
		succeed(cuModuleGetGlobal(&pc_positions, &constant_size, module, "c_positions"));
		succeed(cuModuleGetGlobal(&pc_normals, &constant_size, module, "c_normals"));
		succeed(cuModuleGetGlobal(&pc_texCoords, &constant_size, module, "c_texCoords"));
		succeed(cuModuleGetGlobal(&pc_indices, &constant_size, module, "c_indices"));
		succeed(cuModuleGetGlobal(&pc_patchData, &constant_size, module, "c_patchData"));

		//succeed(cuModuleGetFunction(&kernel_init_vstorage, module, "initVstorageSimpleVertex"));
	}

	void IndexedGeometry::draw() const
	{
		// bind buffers
		CUdeviceptr pointer = vertexBuffer;
		succeed(cuMemcpyHtoD(pc_positions, &pointer, sizeof(CUdeviceptr)));
		pointer = normalBuffer;
		succeed(cuMemcpyHtoD(pc_normals, &pointer, sizeof(CUdeviceptr)));
		pointer = texcoordBuffer;
		succeed(cuMemcpyHtoD(pc_texCoords, &pointer, sizeof(CUdeviceptr)));
		pointer = indexBuffer;
		succeed(cuMemcpyHtoD(pc_indices, &pointer, sizeof(CUdeviceptr)));
		pointer = patchBuffer;
		succeed(cuMemcpyHtoD(pc_patchData, &pointer, sizeof(CUdeviceptr)));

		// clear vertex storage
		//succeed(cuLaunchKernel(kernel_init_vstorage, 1, 1, 1, 1, 1, 1, 0, 0, nullptr, nullptr));
		succeed(cuCtxSynchronize());
	}

	void IndexedGeometry::draw(int start, int num_indices) const
	{
	}


	ClipspaceGeometry::ClipspaceGeometry(Renderer& renderer, CUmodule module, const float* position, size_t num_vertices)
	    : vertexBuffer(CU::allocMemory(num_vertices * 16U)),
	      indexBuffer(CU::allocMemory(num_vertices * 4U)),
	      num_vertices(num_vertices),
	      num_indices(num_vertices),
	      renderer(renderer),
	      module(module)
	{
		std::vector<uint32_t> indices(num_vertices);
		for (int i = 0; i < num_vertices; i++)
		{
			indices[i] = i;
		}

		// generate buffers and copy data to the GPU
		succeed(cuMemcpyHtoD(vertexBuffer, position, num_vertices * 16u));
		succeed(cuMemcpyHtoD(indexBuffer, indices.data(), num_indices * 4u));

		//// if texcoord buffer is zero, there are no tex coords
		//if (texcoord)
		//{
		//    texcoordBuffer = CU::allocMemory(num_vertices * 8u);
		//    succeed(cuMemcpyHtoD(texcoordBuffer, texcoord, num_vertices * 8u));
		//}

		// generate patches
		/*std::vector<std::uint32_t> patchStarts;
		unsigned int max_patch_vertices = GPM_PATCH_MAX_VERTICES;
		if (GEOMETRY_PROCESSING == GPM_WARP_VOTING)
			max_patch_vertices = 32;
		patchStarts.reserve(2 * std::max(num_vertices / max_patch_vertices, num_indices / max_patch_vertices));
		std::set<std::uint32_t> seenIndices;

		patchStarts.push_back(0u);
		std::uint32_t currentPatchIndices = 0u;
		std::uint32_t i;
		for (i = 0u; i + 3u <= num_indices; i += 3u, currentPatchIndices += 3u)
		{
			bool restartPatch = false;

			// if the current triangle goes beyond the max index size -> finish patch
			if (currentPatchIndices + 3 > max_patch_vertices)
				restartPatch = true;
			else
			{
				// can we add the triangle while staying within the limit?
				std::uint32_t toAdd = 0;
				std::for_each(indices.data() + i, indices.data() + i + 3, [&](std::uint32_t id) {
					toAdd += seenIndices.find(id) == seenIndices.end() ? 1 : 0;
				});
				if (seenIndices.size() + toAdd >= max_patch_vertices)
					restartPatch = true;
			}
			if (restartPatch)
			{
				patchStarts.push_back(i);
				currentPatchIndices = 0;
				seenIndices.clear();
			}
			std::for_each(indices.data() + i, indices.data() + i + 3, [&](std::uint32_t id) { seenIndices.insert(id); });
		}
		patchStarts.push_back(i);

		// allocate patch buffer and copy patch data
		patchBuffer = CU::allocMemory(patchStarts.size() * 4u);
		succeed(cuMemcpyHtoD(patchBuffer, &patchStarts[0], patchStarts.size() * 4u));
		num_patches = patchStarts.size() - 1;*/

		// get constants
		size_t constant_size;
		succeed(cuModuleGetGlobal(&pc_positions, &constant_size, module, "c_positions"));
		//succeed(cuModuleGetGlobal(&pc_normals, &constant_size, module, "c_normals"));
		//succeed(cuModuleGetGlobal(&pc_texCoords, &constant_size, module, "c_texCoords"));
		succeed(cuModuleGetGlobal(&pc_indices, &constant_size, module, "c_indices"));
		succeed(cuModuleGetGlobal(&pc_patchData, &constant_size, module, "c_patchData"));

		//succeed(cuModuleGetFunction(&kernel_init_vstorage, module, "initVstorageSimpleVertex"));
	}

	void ClipspaceGeometry::draw() const
	{
		// bind buffers
		CUdeviceptr pointer = vertexBuffer;
		succeed(cuMemcpyHtoD(pc_positions, &pointer, sizeof(CUdeviceptr)));
		//pointer = normalBuffer;
		//succeed(cuMemcpyHtoD(pc_normals, &pointer, sizeof(CUdeviceptr)));
		//pointer = texcoordBuffer;
		//succeed(cuMemcpyHtoD(pc_texCoords, &pointer, sizeof(CUdeviceptr)));
		pointer = indexBuffer;
		succeed(cuMemcpyHtoD(pc_indices, &pointer, sizeof(CUdeviceptr)));
		pointer = patchBuffer;
		succeed(cuMemcpyHtoD(pc_patchData, &pointer, sizeof(CUdeviceptr)));

		// clear vertex storage
		//succeed(cuLaunchKernel(kernel_init_vstorage, 1, 1, 1, 1, 1, 1, 0, 0, nullptr, nullptr));
		succeed(cuCtxSynchronize());
	}

	void ClipspaceGeometry::draw(int start, int num_indices) const
	{
	}
}
