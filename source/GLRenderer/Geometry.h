


#ifndef INCLUDED_GLRENDERER_GEOMETRY
#define INCLUDED_GLRENDERER_GEOMETRY

#pragma once

#include <cstdint>

#include <GL/buffer.h>
#include <GL/vertex_array.h>

#include <Resource.h>


namespace GLRenderer
{
	class Renderer;

	class IndexedTriangles : public ::Geometry
	{
	protected:
		IndexedTriangles(const IndexedTriangles&) = delete;
		IndexedTriangles& operator=(const IndexedTriangles&) = delete;

		GL::VertexArray vao;
		GL::Buffer vbo;
		GL::Buffer ibo;

		GLsizei num_indices;

		Renderer& renderer;

	public:
		IndexedTriangles(Renderer& renderer, const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices);
		void draw() const override;
		void draw(int start, int num_indices) const override;
	};

	class IndexedQuads : public ::Geometry
	{
	protected:
		IndexedQuads(const IndexedQuads&) = delete;
		IndexedQuads& operator=(const IndexedQuads&) = delete;

		GL::VertexArray vao;
		GL::Buffer vbo;
		GL::Buffer ibo;

		GLsizei num_indices;

		Renderer& renderer;

	public:
		IndexedQuads(Renderer& renderer, const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices);
		void draw() const override;
		void draw(int start, int num_indices) const override;
	};

	class ClipspaceGeometry : public ::Geometry
	{
	protected:
		ClipspaceGeometry(const ClipspaceGeometry&) = delete;
		ClipspaceGeometry& operator=(const ClipspaceGeometry&) = delete;

		GL::VertexArray vao;
		GL::Buffer vbo;

		GLsizei num_vertices;

		Renderer& renderer;

	public:
		ClipspaceGeometry(Renderer& renderer, const float* position, size_t num_vertices);
		void draw() const override;
		void draw(int start, int num_indices) const override;
	};

	class EyeCandyGeometry : public ::Geometry
	{
	protected:
		EyeCandyGeometry(const EyeCandyGeometry&) = delete;
		EyeCandyGeometry& operator=(const EyeCandyGeometry&) = delete;

		GLuint element_buffer;
		GL::VertexArray vao;
		GL::Buffer vbo;

		GLsizei num_vertices;
		GLsizei num_triangles;

		Renderer& renderer;

	public:
		EyeCandyGeometry(Renderer& renderer, const float* position, size_t num_vertices, const uint32_t* indices, const float* triangle_colors, size_t num_triangles);

		void draw() const override;
		void draw(int start, int num_indices) const override;
	};
}

#endif // INCLUDED_GLRENDERER_GEOMETRY
