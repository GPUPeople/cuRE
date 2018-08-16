


#include <memory>

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

		auto buffer = std::make_unique<std::uint32_t[]>(num_vertices);

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
		auto buffer = std::make_unique<float[]>(num_vertices * 8);

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
		auto buffer = std::make_unique<float[]>(num_vertices * 8);

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


	WaterDemo::WaterDemo(Pipeline& pipeline, const float* position, size_t num_vertices, const std::uint32_t* indices, size_t num_indices, float* img_data, uint32_t width, uint32_t height, char* normal_data, uint32_t n_width, uint32_t n_height, uint32_t n_levels)
	    : vertex_buffer(CU::allocMemory(num_vertices * 4 * 4U)),
	      index_buffer(CU::allocMemory(num_indices * 4U)),
	      num_vertices(num_vertices),
	      num_indices(num_indices),
	      width(width),
	      height(height),
	      pipeline(pipeline)
	{
		auto buffer = std::make_unique<float[]>(num_vertices * 4);

		for (float* dest = &buffer[0]; dest < &buffer[0] + num_vertices * 4;)
		{
			*dest++ = *position++;
			*dest++ = *position++;
			*dest++ = *position++;
			*dest++ = 0.0f;
		}

		succeed(cuMemcpyHtoD(vertex_buffer, buffer.get(), num_vertices * 4 * 4U));
		succeed(cuMemcpyHtoD(index_buffer, indices, num_indices * 4U));

		auto col_buffer = std::make_unique<float[]>(width * height * 4);
		for (unsigned int i = 0; i < width * height; i++)
		{
			col_buffer[i * 4 + 0] = img_data[i * 3 + 0];
			col_buffer[i * 4 + 1] = img_data[i * 3 + 1];
			col_buffer[i * 4 + 2] = img_data[i * 3 + 2];
			col_buffer[i * 4 + 3] = 1.0f;
		}

		{
			color_array = CU::createArray2D(width, height, CU_AD_FORMAT_FLOAT, 4);

			CUDA_MEMCPY2D cpy_info;
			cpy_info.dstMemoryType = CU_MEMORYTYPE_ARRAY;
			cpy_info.dstArray = color_array;
			cpy_info.dstXInBytes = 0;
			cpy_info.dstY = 0;

			cpy_info.srcMemoryType = CU_MEMORYTYPE_HOST;
			cpy_info.srcHost = col_buffer.get();
			cpy_info.srcPitch = width * sizeof(float) * 4;
			cpy_info.srcXInBytes = 0;
			cpy_info.srcY = 0;

			cpy_info.WidthInBytes = width * sizeof(float) * 4;
			cpy_info.Height = height;
			succeed(cuMemcpy2D(&cpy_info));

			pipeline.setTextureF(color_array);
		}

		{
			texi = CU::createArray2DMipmapped(n_width, n_height, n_levels, CU_AD_FORMAT_UNSIGNED_INT8, 4);

			for (unsigned int i = 0; i < n_levels; ++i)
			{
				CUarray array;
				succeed(cuMipmappedArrayGetLevel(&array, texi, i));

				CUDA_MEMCPY3D cpy;

				cpy.srcXInBytes = 0U;
				cpy.srcY = 0U;
				cpy.srcZ = 0U;
				cpy.srcLOD = 0U;
				cpy.srcMemoryType = CU_MEMORYTYPE_HOST;
				cpy.srcHost = normal_data;
				cpy.srcPitch = n_width * 4U;
				cpy.srcHeight = n_height;

				cpy.dstXInBytes = 0U;
				cpy.dstY = 0U;
				cpy.dstZ = 0U;
				cpy.dstLOD = 0U;
				cpy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
				cpy.dstArray = array;

				cpy.WidthInBytes = n_width * 4U;
				cpy.Height = n_height;
				cpy.Depth = 1;

				succeed(cuMemcpy3D(&cpy));

				normal_data += (n_width * n_height * 4);

				n_width = std::max<unsigned int>(n_width / 2, 1);
				n_height = std::max<unsigned int>(n_height / 2, 1);
			}

			pipeline.setTexture(texi, std::numeric_limits<float>::max());
		}
	}

	void WaterDemo::draw() const
	{
		pipeline.drawWaterDemo(vertex_buffer, num_vertices, index_buffer, num_indices);
	}

	void WaterDemo::draw(int from, int num_indices) const
	{
	}


	BlendGeometry::BlendGeometry(Pipeline& pipeline, const float* position, const float* normals, const float* color, size_t num_vertices)
	    : vertex_buffer(CU::allocMemory(num_vertices * 8 * sizeof(float))),
	      index_buffer(CU::allocMemory(num_vertices * 3 * sizeof(uint32_t))),
	      num_vertices(num_vertices),
	      pipeline(pipeline)
	{
		auto vertices = std::make_unique<float[]>(num_vertices * 8);

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

		auto indices = std::make_unique<uint32_t[]>(num_vertices);
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
