


#include <memory>

#include <core/utils/memory>

#include <CUDA/error.h>
#include <CUDA/memory.h>

#include <math/vector.h>

#include "utils.h"

#include "Geometry.h"
#include "Pipeline.h"
#include <cstdint>
#include <iostream>


namespace cuRE
{
	ClipspaceGeometry::ClipspaceGeometry(Pipeline& pipeline, const float* position, size_t num_vertices)
		: vertex_buffer(CU::allocMemory(num_vertices * 4 * 4U)),
		  index_buffer(CU::allocMemory(num_vertices * 4U)),
		  num_vertices(num_vertices),
		  pipeline(pipeline)
	{
		succeed(cuMemcpyHtoD(vertex_buffer, position, num_vertices * 4 * 4U));

		auto buffer = core::make_unique_default<std::uint32_t[]>(num_vertices);

		for (std::uint32_t i = 0U; i < num_vertices; ++i)
			buffer[i] = i;

		succeed(cuMemcpyHtoD(index_buffer, &buffer[0], num_vertices * 4U));
	}

	void ClipspaceGeometry::draw() const
	{
		pipeline.drawTriangles(vertex_buffer, num_vertices, index_buffer, num_vertices);
	}

	void ClipspaceGeometry::draw(int from, int num_indices) const
	{
	}


	IndexedTriangles::IndexedTriangles(Pipeline& pipeline, const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices)
	    : vertex_buffer(CU::allocMemory(num_vertices * 8 * 4U)),
	      index_buffer(CU::allocMemory(num_indices * 4U)),
	      num_vertices(num_vertices),
	      num_indices(num_indices),
	      pipeline(pipeline)
	{
		auto buffer = core::make_unique_default<float[]>(num_vertices * 8);

		for (float* dest = &buffer[0]; dest < &buffer[0] + num_vertices * 8;)
		{
			*dest++ = *position++;
			*dest++ = *position++;
			*dest++ = *position++;

			*dest++ = *normals++;
			*dest++ = *normals++;
			*dest++ = *normals++;

			*dest++ = *texcoord++;
			*dest++ = *texcoord++;
		}

		succeed(cuMemcpyHtoD(vertex_buffer, &buffer[0], num_vertices * 8 * 4U));

		succeed(cuMemcpyHtoD(index_buffer, indices, num_indices * 4U));
	}

	void IndexedTriangles::draw() const
	{
		pipeline.drawTriangles(vertex_buffer, num_vertices, index_buffer, num_indices);
	}

	void IndexedTriangles::draw(int from, int num_indices) const
	{
	}


	IndexedQuads::IndexedQuads(Pipeline& pipeline, const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices)
	    : vertex_buffer(CU::allocMemory(num_vertices * 8 * 4U)),
	      index_buffer(CU::allocMemory(num_indices * 4U)),
	      num_vertices(num_vertices),
	      num_indices(num_indices),
	      pipeline(pipeline)
	{
		auto buffer = core::make_unique_default<float[]>(num_vertices * 8);

		for (float* dest = &buffer[0]; dest < &buffer[0] + num_vertices * 8;)
		{
			*dest++ = *position++;
			*dest++ = *position++;
			*dest++ = *position++;

			*dest++ = *normals++;
			*dest++ = *normals++;
			*dest++ = *normals++;

			*dest++ = *texcoord++;
			*dest++ = *texcoord++;
		}

		succeed(cuMemcpyHtoD(vertex_buffer, &buffer[0], num_vertices * 8 * 4U));

		succeed(cuMemcpyHtoD(index_buffer, indices, num_indices * 4U));
	}

	void IndexedQuads::draw() const
	{
		pipeline.drawQuads(vertex_buffer, num_vertices, index_buffer, num_indices);
	}

	void IndexedQuads::draw(int from, int num_indices) const
	{
	}


	EyeCandyGeometry::EyeCandyGeometry(Pipeline& pipeline, const float* vertices, size_t num_vertices, const uint32_t* indices, const float* triangle_colors, size_t num_triangles)
		: vertex_buffer(CU::allocMemory(num_vertices * 12 * sizeof(float))),
		  index_buffer(CU::allocMemory(num_triangles * 3 * sizeof(uint32_t))),
		  num_vertices(num_vertices),
		  num_triangles(num_triangles),
		  pipeline(pipeline)
	{
		succeed(cuMemcpyHtoD(vertex_buffer, vertices, num_vertices * 12 * sizeof(float)));
		succeed(cuMemcpyHtoD(index_buffer, indices, num_triangles * 3 * sizeof(uint32_t)));
	}

	void EyeCandyGeometry::draw() const
	{
		pipeline.drawTriangles(vertex_buffer, num_vertices, index_buffer, num_triangles * 3);
	}

	void EyeCandyGeometry::draw(int from, int num_indices) const
	{
		CUdeviceptr temp = index_buffer;
		temp += from * sizeof(uint32_t);
		pipeline.drawTriangles(vertex_buffer, num_vertices, temp, num_indices);
	}


	OceanGeometry::OceanGeometry(Pipeline& pipeline, const float* position, size_t num_vertices, const std::uint32_t* indices, size_t num_indices)
	    : vertex_buffer(CU::allocMemory(num_vertices * 4 * 4U)),
	      index_buffer(CU::allocMemory(num_indices * 4U)),
	      num_vertices(num_vertices),
	      num_indices(num_indices),
	      pipeline(pipeline)
	{
		auto buffer = core::make_unique_default<float[]>(num_vertices * 4);

		for (float* dest = &buffer[0]; dest < &buffer[0] + num_vertices * 4;)
		{
			*dest++ = *position++;
			*dest++ = *position++;
			*dest++ = *position++;
			*dest++ = 0.0f;
		}

		succeed(cuMemcpyHtoD(vertex_buffer, buffer.get(), num_vertices * 4 * 4U));
		succeed(cuMemcpyHtoD(index_buffer, indices, num_indices * 4U));
	}

	void OceanGeometry::draw() const
	{
	}

	void OceanGeometry::draw(int from, int num_indices) const
	{
		//                                                                                massive       HACK!!!
		pipeline.drawOcean(vertex_buffer, num_vertices, index_buffer, this->num_indices, from != 0, num_indices != 0);
	}


	BlendGeometry::BlendGeometry(Pipeline& pipeline, const float* position, const float* normals, const float* color, size_t num_vertices)
	    : vertex_buffer(CU::allocMemory(num_vertices * 8 * sizeof(float))),
	      index_buffer(CU::allocMemory(num_vertices * 3 * sizeof(uint32_t))),
	      num_vertices(num_vertices),
	      pipeline(pipeline)
	{
		auto vertices = core::make_unique_default<float[]>(num_vertices * 8);

		for (float* dest = &vertices[0]; dest < &vertices[0] + num_vertices * 8;)
		{
			*dest++ = *position++;
			*dest++ = *position++;

			*dest++ = *normals++;
			*dest++ = *normals++;
			*dest++ = *normals++;

			*dest++ = *color++;
			*dest++ = *color++;
			*dest++ = *color++;
		}

		succeed(cuMemcpyHtoD(vertex_buffer, vertices.get(), num_vertices * 8 * sizeof(float)));

		auto indices = core::make_unique_default<uint32_t[]>(num_vertices);
		for (int i = 0; i < num_vertices; i++)
		{
			indices[i] = i;
		}

		succeed(cuMemcpyHtoD(index_buffer, indices.get(), num_vertices * sizeof(uint32_t)));
	}

	void BlendGeometry::draw() const
	{
		pipeline.drawBlendDemo(vertex_buffer, num_vertices, index_buffer, num_vertices);
	}

	void BlendGeometry::draw(int from, int num_indices) const
	{
	}


	IsoBlendGeometry::IsoBlendGeometry(Pipeline& pipeline, const float* vertex_data, uint32_t num_vertices, const uint32_t* index_data, uint32_t num_indices)
	    : vertex_buffer(CU::allocMemory(num_vertices * 12 * sizeof(float))),
	      index_buffer(CU::allocMemory(num_indices * 3 * sizeof(uint32_t))),
	      num_indices(num_indices),
	      num_vertices(num_vertices),
	      pipeline(pipeline)
	{
		std::vector<float> tempsi(12 * num_vertices);

		for (unsigned int i = 0; i < num_vertices; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				tempsi[i * 12 + j] = vertex_data[i * 10 + j];
			}
		}

		succeed(cuMemcpyHtoD(vertex_buffer, tempsi.data(), num_vertices * 12 * sizeof(float)));
		succeed(cuMemcpyHtoD(index_buffer, index_data, num_indices * 3 * sizeof(uint32_t)));
	}

	void IsoBlendGeometry::draw() const
	{
		pipeline.drawIsoBlendDemo(vertex_buffer, num_vertices, index_buffer, num_indices * 3);
	}

	void IsoBlendGeometry::draw(int from, int num_indices) const
	{
	}


	GlyphGeometry::GlyphGeometry(Pipeline& pipeline, uint64_t mask, const float* vertex_data, uint32_t num_vertices, const uint32_t* index_data, uint32_t num_indices)
	    : vertex_buffer(CU::allocMemory(num_vertices * 12 * sizeof(float))),
	      index_buffer(CU::allocMemory(num_indices * 3 * sizeof(uint32_t))),
	      num_indices(num_indices),
	      num_vertices(num_vertices),
	      pipeline(pipeline),
	      mask(mask)
	{
		std::vector<float> tempsi(12 * num_vertices);

		for (unsigned int i = 0; i < num_vertices; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				tempsi[i * 12 + j] = vertex_data[i * 10 + j];
			}
		}

		succeed(cuMemcpyHtoD(vertex_buffer, tempsi.data(), num_vertices * 12 * sizeof(float)));
		succeed(cuMemcpyHtoD(index_buffer, index_data, num_indices * 3 * sizeof(uint32_t)));
	}

	void GlyphGeometry::draw() const
	{
		pipeline.setStipple(mask);
		pipeline.drawGlyphDemo(vertex_buffer, num_vertices, index_buffer, num_indices * 3);
	}

	void GlyphGeometry::draw(int from, int num_indices) const
	{
	}


	IsoStippleGeometry::IsoStippleGeometry(Pipeline& pipeline, uint64_t mask, const float* vertex_data, uint32_t num_vertices, const uint32_t* index_data, uint32_t num_indices)
	    : vertex_buffer(CU::allocMemory(num_vertices * 12 * sizeof(float))),
	      index_buffer(CU::allocMemory(num_indices * 3 * sizeof(uint32_t))),
	      num_indices(num_indices),
	      num_vertices(num_vertices),
	      pipeline(pipeline),
	      mask(mask)
	{
		std::vector<float> tempsi(12 * num_vertices);

		for (unsigned int i = 0; i < num_vertices; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				tempsi[i * 12 + j] = vertex_data[i * 10 + j];
			}
		}

		succeed(cuMemcpyHtoD(vertex_buffer, tempsi.data(), num_vertices * 12 * sizeof(float)));
		succeed(cuMemcpyHtoD(index_buffer, index_data, num_indices * 3 * sizeof(uint32_t)));
	}

	void IsoStippleGeometry::draw() const
	{
		pipeline.setStipple(mask);
		pipeline.drawIsoStipple(vertex_buffer, num_vertices, index_buffer, num_indices * 3);
	}

	void IsoStippleGeometry::draw(int from, int num_indices) const
	{
	}


	CheckerboardGeometry::CheckerboardGeometry(Pipeline& pipeline, int type, const float* vertices, size_t num_vertices, const uint32_t* indices, const float* triangle_colors, size_t num_triangles)
	    : vertex_buffer(CU::allocMemory(num_vertices * 12 * sizeof(float))),
	      index_buffer(CU::allocMemory(num_triangles * 3 * sizeof(uint32_t))),
	      num_vertices(num_vertices),
	      num_triangles(num_triangles),
	      pipeline(pipeline),
	      type(type)
	{
		succeed(cuMemcpyHtoD(vertex_buffer, vertices, num_vertices * 12 * sizeof(float)));
		succeed(cuMemcpyHtoD(index_buffer, indices, num_triangles * 3 * sizeof(uint32_t)));
	}

	void CheckerboardGeometry::draw() const
	{
		if (((type >> 1) & 0x1U) == 0U)
		{
			if ((type & 0x1U) == 0U)
				pipeline.drawCheckerboardGeometry(vertex_buffer, num_vertices, index_buffer, num_triangles * 3);
			else
				pipeline.drawCheckerboardFragmentGeometry(vertex_buffer, num_vertices, index_buffer, num_triangles * 3);
		}
		else
		{
			if ((type & 0x1U) == 0U)
				pipeline.drawCheckerboardQuadGeometry(vertex_buffer, num_vertices, index_buffer, num_triangles * 3);
			else
				pipeline.drawCheckerboardQuadFragmentGeometry(vertex_buffer, num_vertices, index_buffer, num_triangles * 3);
		}
	}

	void CheckerboardGeometry::draw(int from, int num_indices) const
	{
	}
}
