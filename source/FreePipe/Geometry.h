


#ifndef INCLUDED_FREEPIPE_GEOMETRY
#define INCLUDED_FREEPIPE_GEOMETRY

#include <cstdint>

#include <CUDA/memory.h>
#include <CUDA/module.h>

#include <Resource.h>


namespace FreePipe
{
	class Renderer;

	class IndexedGeometry : public ::Geometry
	{
	protected:
		IndexedGeometry(const IndexedGeometry&) = delete;
		IndexedGeometry& operator=(const IndexedGeometry&) = delete;

		CU::unique_ptr vertexBuffer;
		CU::unique_ptr normalBuffer;
		CU::unique_ptr texcoordBuffer;
		CU::unique_ptr indexBuffer;
		CU::unique_ptr patchBuffer;

		CUdeviceptr pc_positions, pc_normals, pc_texCoords;
		CUdeviceptr pc_indices, pc_patchData;

		CUmodule module;
		//CUfunction kernel_init_vstorage;

		size_t num_vertices;
		size_t num_indices;
		size_t num_patches;

		Renderer& renderer;

	public:
		IndexedGeometry(Renderer& renderer, CUmodule module, const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices);

		size_t getNumVertices() const { return num_vertices; }
		size_t getNumTriangles() const { return num_indices / 3; }
		size_t getNumPatches() const { return num_patches; }

		void draw() const override;
		void draw(int start, int num_indices) const override;
	};

	class ClipspaceGeometry : public ::Geometry
	{
	protected:
		ClipspaceGeometry(const ClipspaceGeometry&) = delete;
		ClipspaceGeometry& operator=(const ClipspaceGeometry&) = delete;

		CU::unique_ptr vertexBuffer;
		CU::unique_ptr normalBuffer;
		CU::unique_ptr texcoordBuffer;
		CU::unique_ptr indexBuffer;
		CU::unique_ptr patchBuffer;

		CUdeviceptr pc_positions, pc_normals, pc_texCoords;
		CUdeviceptr pc_indices, pc_patchData;

		CUmodule module;
		//CUfunction kernel_init_vstorage;

		size_t num_vertices;
		size_t num_indices;
		size_t num_patches;

		Renderer& renderer;

	public:
		ClipspaceGeometry(Renderer& renderer, CUmodule module, const float* position, size_t num_vertices);

		size_t getNumVertices() const { return num_vertices; }
		size_t getNumTriangles() const { return num_indices / 3; }
		size_t getNumPatches() const { return num_patches; }

		void draw() const override;
		void draw(int start, int num_indices) const override;
	};
}

#endif // INCLUDED_FREEPIPE_GEOMETRY
