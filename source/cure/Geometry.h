
#ifndef INCLUDED_CURE_GEOMETRY
#define INCLUDED_CURE_GEOMETRY

#include <cstdint>

#include <CUDA/array.h>
#include <CUDA/memory.h>
#include <CUDA/module.h>
#include <Resource.h>
namespace cuRE
{
	class Pipeline;

	class ClipspaceGeometry : public ::Geometry
	{
	protected:
		ClipspaceGeometry(const ClipspaceGeometry&) = delete;
		ClipspaceGeometry& operator=(const ClipspaceGeometry&) = delete;

		CU::unique_ptr vertex_buffer;
		CU::unique_ptr index_buffer;

		size_t num_vertices;

		Pipeline& pipeline;

	public:
		ClipspaceGeometry(Pipeline& pipeline, const float* position, size_t num_vertices);

		void draw() const override;
		void draw(int from, int num_indices) const override;
	};

	class IndexedTriangles : public ::Geometry
	{
	protected:
		IndexedTriangles(const IndexedTriangles&) = delete;
		IndexedTriangles& operator=(const IndexedTriangles&) = delete;

		CU::unique_ptr vertex_buffer;
		CU::unique_ptr index_buffer;

		size_t num_vertices;
		size_t num_indices;

		Pipeline& pipeline;

	public:
		IndexedTriangles(Pipeline& pipeline, const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices);

		void draw() const override;
		void draw(int start, int num_indices) const override;
	};

	class IndexedQuads : public ::Geometry
	{
	protected:
		IndexedQuads(const IndexedQuads&) = delete;
		IndexedQuads& operator=(const IndexedQuads&) = delete;

		CU::unique_ptr vertex_buffer;
		CU::unique_ptr index_buffer;

		size_t num_vertices;
		size_t num_indices;

		Pipeline& pipeline;

	public:
		IndexedQuads(Pipeline& pipeline, const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices);

		void draw() const override;
		void draw(int start, int num_indices) const override;
	};

	class EyeCandyGeometry : public ::Geometry
	{
	protected:
		EyeCandyGeometry(const EyeCandyGeometry&) = delete;
		EyeCandyGeometry& operator=(const EyeCandyGeometry&) = delete;

		CU::unique_ptr vertex_buffer;
		CU::unique_ptr index_buffer;
		CU::unique_ptr color_buffer;

		size_t num_vertices;
		size_t num_triangles;

		Pipeline& pipeline;

	public:
		EyeCandyGeometry(Pipeline& pipeline, const float* position, size_t num_vertices, const uint32_t* indices, const float* triangle_colors, size_t num_triangles);

		void draw() const override;
		void draw(int from, int num_indices) const override;
	};

	class WaterDemo : public ::Geometry
	{
	protected:
		WaterDemo(const WaterDemo&) = delete;
		WaterDemo& operator=(const WaterDemo&) = delete;

		CU::unique_ptr vertex_buffer;
		CU::unique_ptr index_buffer;
		CU::unique_array color_array;
		CU::unique_array normal_array;
		//CUtexObject tex;
		CU::unique_mipmapped_array texi;

		size_t num_vertices;
		size_t num_indices;
		uint32_t width;
		uint32_t height;

		Pipeline& pipeline;

	public:
		WaterDemo(Pipeline& pipeline, const float* position, size_t num_vertices, const std::uint32_t* indices, size_t num_indices, float* img_data, uint32_t width, uint32_t height, char* normal_data, uint32_t n_width, uint32_t n_height, uint32_t n_levels);

		void draw() const override;
		void draw(int start, int num_indices) const override;
	};

	class BlendGeometry : public ::Geometry
	{
	protected:
		BlendGeometry(const BlendGeometry&) = delete;
		BlendGeometry& operator=(const BlendGeometry&) = delete;

		CU::unique_ptr vertex_buffer;
		CU::unique_ptr index_buffer;

		size_t num_vertices;

		Pipeline& pipeline;

	public:
		BlendGeometry(Pipeline& pipeline, const float* position, const float* normals, const float* color, size_t num_vertices);

		void draw() const override;
		void draw(int start, int num_indices) const override;
	};

	class IsoBlendGeometry : public ::Geometry
	{
	protected:
		IsoBlendGeometry(const IsoBlendGeometry&) = delete;
		IsoBlendGeometry& operator=(const IsoBlendGeometry&) = delete;

		CU::unique_ptr vertex_buffer;
		CU::unique_ptr index_buffer;

		size_t num_indices;
		size_t num_vertices;

		Pipeline& pipeline;

	public:
		IsoBlendGeometry(Pipeline& pipeline, const float* vertex_data, uint32_t num_vertices, const uint32_t* index_data, uint32_t num_indices);

		void draw() const override;
		void draw(int start, int num_indices) const override;
	};

	class GlyphGeometry : public ::Geometry
	{
	protected:
		GlyphGeometry(const GlyphGeometry&) = delete;
		GlyphGeometry& operator=(const GlyphGeometry&) = delete;

		CU::unique_ptr vertex_buffer;
		CU::unique_ptr index_buffer;

		size_t num_indices;
		size_t num_vertices;

		uint64_t mask;

		Pipeline& pipeline;

	public:
		GlyphGeometry(Pipeline& pipeline, uint64_t mask, const float* vertex_data, uint32_t num_vertices, const uint32_t* index_data, uint32_t num_indices);

		void draw() const override;
		void draw(int start, int num_indices) const override;
	};

	class IsoStippleGeometry : public ::Geometry
	{
	protected:
		IsoStippleGeometry(const IsoStippleGeometry&) = delete;
		IsoStippleGeometry& operator=(const IsoStippleGeometry&) = delete;

		CU::unique_ptr vertex_buffer;
		CU::unique_ptr index_buffer;

		size_t num_indices;
		size_t num_vertices;

		Pipeline& pipeline;

		uint64_t mask;

	public:
		IsoStippleGeometry(Pipeline& pipeline, uint64_t mask, const float* vertex_data, uint32_t num_vertices, const uint32_t* index_data, uint32_t num_indices);

		void draw() const override;
		void draw(int start, int num_indices) const override;
	};

	class CheckerboardGeometry : public ::Geometry
	{
	protected:
		CheckerboardGeometry(const CheckerboardGeometry&) = delete;
		CheckerboardGeometry& operator=(const CheckerboardGeometry&) = delete;

		CU::unique_ptr vertex_buffer;
		CU::unique_ptr index_buffer;
		CU::unique_ptr color_buffer;

		size_t num_vertices;
		size_t num_triangles;

		Pipeline& pipeline;

		int type;

	public:
		CheckerboardGeometry(Pipeline& pipeline, int type, const float* position, size_t num_vertices, const uint32_t* indices, const float* triangle_colors, size_t num_triangles);

		void draw() const override;
		void draw(int start, int num_indices) const override;
	};
}

#endif // INCLUDED_CURE_GEOMETRY
