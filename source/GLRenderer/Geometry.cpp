


#include <cstring>
#include <memory>

#include <GL/error.h>

#include "Geometry.h"
#include "Renderer.h"


namespace
{
	auto constructVertexBuffer(const float* position, const float* normals, const float* texcoord, size_t num_vertices)
	{
		auto buffer_data = std::unique_ptr<float[]>{new float[8U * num_vertices]};

		for (float* dest = &buffer_data[0]; dest < &buffer_data[0] + 8U * num_vertices; dest += 8)
		{
			dest[0] = position[0];
			dest[1] = position[1];
			dest[2] = position[2];
			dest[3] = normals[0];
			dest[4] = normals[1];
			dest[5] = normals[2];
			dest[6] = texcoord[0];
			dest[7] = texcoord[1];
			position += 3;
			normals += 3;
			texcoord += 2;
		}

		return buffer_data;
	}
}

namespace GLRenderer
{
	IndexedTriangles::IndexedTriangles(Renderer& renderer, const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices)
	    : renderer(renderer),
	      num_indices(static_cast<GLsizei>(num_indices))
	{
		{
			auto buffer_data = constructVertexBuffer(position, normals, texcoord, num_vertices);

			glBindVertexArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferStorage(GL_ARRAY_BUFFER, 32U * num_vertices, &buffer_data[0], 0U);
		}

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
		glBufferStorage(GL_ELEMENT_ARRAY_BUFFER, 4U * num_indices, indices, 0U);

		GL::throw_error();

		glBindVertexArray(vao);
		glBindVertexBuffer(0U, vbo, 0U, 32U);
		glVertexAttribFormat(0U, 3, GL_FLOAT, GL_FALSE, 0U);
		glVertexAttribFormat(1U, 3, GL_FLOAT, GL_FALSE, 12U);
		glVertexAttribFormat(2U, 2, GL_FLOAT, GL_FALSE, 24U);
		glVertexAttribBinding(0U, 0U);
		glVertexAttribBinding(1U, 0U);
		glVertexAttribBinding(2U, 0U);
		glEnableVertexAttribArray(0U);
		glEnableVertexAttribArray(1U);
		glEnableVertexAttribArray(2U);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

		GL::throw_error();
	}

	void IndexedTriangles::draw() const
	{
		glBindVertexArray(vao);
		renderer.beginDrawTiming();
		glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, nullptr);
		renderer.endDrawTiming();
		GL::throw_error();
	}

	void IndexedTriangles::draw(int start, int num_indices) const
	{
		glBindVertexArray(vao);
		renderer.beginDrawTiming();
		glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(start)));
		renderer.endDrawTiming();
		GL::throw_error();
	}

	IndexedQuads::IndexedQuads(Renderer& renderer, const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices)
	    : renderer(renderer),
	      num_indices(static_cast<GLsizei>(6U * num_indices / 4U))
	{
		{
			auto buffer_data = constructVertexBuffer(position, normals, texcoord, num_vertices);

			glBindVertexArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferStorage(GL_ARRAY_BUFFER, 32U * num_vertices, &buffer_data[0], 0U);

			GL::throw_error();
		}

		{
			auto buffer_data = std::make_unique<std::uint32_t[]>(this->num_indices);

			const std::uint32_t* src = indices;
			std::uint32_t* dest = &buffer_data[0];

			while (src < indices + num_indices)
			{
				*dest++ = src[1];
				*dest++ = src[2];
				*dest++ = src[0];
				*dest++ = src[0];
				*dest++ = src[2];
				*dest++ = src[3];
				src += 4;
			}

			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
			glBufferStorage(GL_ELEMENT_ARRAY_BUFFER, 4U * this->num_indices, &buffer_data[0], 0U);

			GL::throw_error();
		}

		glBindVertexArray(vao);
		glBindVertexBuffer(0U, vbo, 0U, 32U);
		glVertexAttribFormat(0U, 3, GL_FLOAT, GL_FALSE, 0U);
		glVertexAttribFormat(1U, 3, GL_FLOAT, GL_FALSE, 12U);
		glVertexAttribFormat(2U, 2, GL_FLOAT, GL_FALSE, 24U);
		glVertexAttribBinding(0U, 0U);
		glVertexAttribBinding(1U, 0U);
		glVertexAttribBinding(2U, 0U);
		glEnableVertexAttribArray(0U);
		glEnableVertexAttribArray(1U);
		glEnableVertexAttribArray(2U);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

		GL::throw_error();
	}

	void IndexedQuads::draw() const
	{
		glBindVertexArray(vao);
		renderer.beginDrawTiming();
		glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, nullptr);
		renderer.endDrawTiming();
		GL::throw_error();
	}

	void IndexedQuads::draw(int start, int num_indices) const
	{
		glBindVertexArray(vao);
		renderer.beginDrawTiming();
		glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(start)));
		renderer.endDrawTiming();
		GL::throw_error();
	}


	ClipspaceGeometry::ClipspaceGeometry(Renderer& renderer, const float* position, size_t num_vertices)
	    : renderer(renderer),
	      num_vertices(static_cast<GLsizei>(num_vertices))
	{
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferStorage(GL_ARRAY_BUFFER, 16U * num_vertices, position, 0U);

		glBindVertexArray(vao);
		glBindVertexBuffer(0U, vbo, 0U, 16U);
		glVertexAttribFormat(0U, 4, GL_FLOAT, GL_FALSE, 0U);
		glVertexAttribBinding(0U, 0U);
		glEnableVertexAttribArray(0U);

		GL::throw_error();
	}

	void ClipspaceGeometry::draw() const
	{
		glBindVertexArray(vao);
		renderer.beginDrawTiming();
		glDrawArrays(GL_TRIANGLES, 0, num_vertices);
		renderer.endDrawTiming();
		GL::throw_error();
	}

	void ClipspaceGeometry::draw(int start, int num_indices) const
	{
	}


	EyeCandyGeometry::EyeCandyGeometry(Renderer& renderer, const float* position, size_t num_vertices, const uint32_t* indices, const float* triangle_colors, size_t num_triangles)
	    : renderer(renderer),
	      num_triangles(static_cast<GLsizei>(num_triangles)),
	      num_vertices(static_cast<GLsizei>(num_vertices))
	{
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferStorage(GL_ARRAY_BUFFER, 48U * num_vertices, position, 0U);

		glBindVertexArray(vao);
		glBindVertexBuffer(0U, vbo, 0U, 48U);
		glVertexAttribFormat(0U, 4, GL_FLOAT, GL_FALSE, 0U);
		glVertexAttribFormat(1U, 4, GL_FLOAT, GL_FALSE, 16U);
		glVertexAttribFormat(2U, 4, GL_FLOAT, GL_FALSE, 32U);
		glVertexAttribBinding(0U, 0U);
		glVertexAttribBinding(1U, 0U);
		glVertexAttribBinding(2U, 0U);
		glEnableVertexAttribArray(0U);
		glEnableVertexAttribArray(1U);
		glEnableVertexAttribArray(2U);

		// Generate a buffer for the indices
		glGenBuffers(1, &element_buffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, num_triangles * 3 * sizeof(uint32_t), &indices[0], GL_STATIC_DRAW);

		GL::throw_error();
	}

	void EyeCandyGeometry::draw() const
	{
		glBindVertexArray(vao);
		renderer.beginDrawTiming();
		glDrawElements(GL_TRIANGLES, num_triangles * 3, GL_UNSIGNED_INT, nullptr);
		renderer.endDrawTiming();
		GL::throw_error();
	}

	void EyeCandyGeometry::draw(int start, int num_indices) const
	{
	}
}
