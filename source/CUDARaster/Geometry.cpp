


#include <stdexcept>

#include "Geometry.h"
#include "Renderer.h"


namespace CUDARaster
{
	IndexedGeometry::IndexedGeometry(Renderer* renderer, const float* position, const float* normals, const float* texcoord, size_t num_vertices, const std::uint32_t* indices, size_t num_indices, FW::App& app)
		: app_(app),
		  num_verts(static_cast<std::uint32_t>(num_vertices)),
		  num_triangles(static_cast<std::uint32_t>(num_indices / 3)),
		  renderer(renderer)
	{
		resizeDiscard(in_vertices, num_vertices * sizeof(FW::InputVertex));
		resizeDiscard(out_vertices, num_vertices * sizeof(FW::ShadedVertex_gouraud));
		resizeDiscard(in_indices, num_indices * sizeof(std::uint32_t));
		
		{
			MutableMem<FW::InputVertex> vertexmem(in_vertices.address, num_vertices);
			FW::InputVertex* v = vertexmem.get();
			for (int i = 0; i < num_vertices; i++)
			{
				v[i].modelPos = FW::Vec3f(position[i*3], position[i*3+1], position[i*3+2]);
				v[i].modelNormal = FW::Vec3f(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
				v[i].texCoord = FW::Vec2f(texcoord[i * 2], texcoord[i * 2 + 1]);
			}
			
			MutableMem<std::uint32_t> indexmem(in_indices.address, num_indices);
			std::uint32_t* i = indexmem.get();
			for(int j = 0; j < num_indices; j += 3)
			{
				i[j+2] = indices[j];
				i[j+1] = indices[j+1];
				i[j] = indices[j+2];
			}
		}
	}

	void IndexedGeometry::draw() const
	{
		Buffer dummy_mat, dummy_vmat, dummy_tmat;
		app_.setData(in_vertices, out_vertices, dummy_mat, in_indices, dummy_vmat, dummy_tmat, num_verts, 0, num_triangles);
		app_.setGlobals();

		renderer->recordDrawingTime(app_.render(0, num_triangles));
	}

	void IndexedGeometry::draw(int from, int num_indices) const
	{
	}


	ClipspaceGeometry::ClipspaceGeometry(Renderer* renderer, const float* position, size_t num_vertices, FW::App& app)
		: app_(app),
		  num_verts(static_cast<std::uint32_t>(num_vertices)),
		  renderer(renderer)
	{
		resizeDiscard(in_vertices, num_vertices * sizeof(FW::ClipSpaceVertex));
		resizeDiscard(out_vertices, num_vertices * sizeof(FW::ShadedVertex_clipSpace));
		resizeDiscard(in_indices, num_vertices * sizeof(std::uint32_t));

		{
			MutableMem<FW::ClipSpaceVertex> vertexmem(in_vertices.address, num_vertices);
			FW::ClipSpaceVertex* v = vertexmem.get();
			for (int i = 0; i < num_vertices; i++)
			{
				v[i].clipSpacePos = FW::Vec4f(position[i * 4 + 0], position[i * 4 + 1], position[i * 4 + 2], position[i * 4 + 3]);
			}

			MutableMem<std::uint32_t> indexmem(in_indices.address, num_vertices);
			std::uint32_t* i = indexmem.get();
			for (int j = 0; j < num_vertices; j++)
			{
				i[j] = j;
			}
		}
	}

	void ClipspaceGeometry::draw() const
	{
		Buffer dummy_mat, dummy_vmat, dummy_tmat;
		app_.setData(in_vertices, out_vertices, dummy_mat, in_indices, dummy_vmat, dummy_tmat, num_verts, 0, num_verts / 3);
		app_.setGlobals();

		renderer->recordDrawingTime(app_.render(0, num_verts / 3));
	}

	void ClipspaceGeometry::draw(int from, int num_indices) const
	{
	}

	EyeCandyGeometry::EyeCandyGeometry(Renderer* renderer, const float* vert_data, size_t num_vertices, const uint32_t* indices, const float* triangle_colors, size_t num_triangles, FW::App& app)
		: app_(app),
		  num_verts(static_cast<std::uint32_t>(num_vertices)),
		  num_indices(static_cast<std::uint32_t>(num_indices)),
		  renderer(renderer)
	{
		resizeDiscard(in_vertices, num_vertices * sizeof(FW::EyecandyVertex));
		resizeDiscard(out_vertices, num_vertices * sizeof(FW::ShadedVertex_eyecandy));
		resizeDiscard(in_indices, num_triangles * 3 * sizeof(std::uint32_t));

		{
			MutableMem<FW::EyecandyVertex> vertexmem(in_vertices.address, num_vertices);
			FW::EyecandyVertex* v = vertexmem.get();
			for (int i = 0; i < num_vertices; i++)
			{
				int off = i * 12;
				v[i].pos = FW::Vec4f(vert_data[off + 0], vert_data[off + 1], vert_data[off + 2], vert_data[off + 3]);
				v[i].normal = FW::Vec4f(vert_data[off + 4], vert_data[off + 5], vert_data[off + 6], vert_data[off + 7]);
				v[i].color = FW::Vec4f(vert_data[off + 8], vert_data[off + 9], vert_data[off + 10], vert_data[off + 11]);
			}

			MutableMem<std::uint32_t> indexmem(in_indices.address, num_triangles * 3);
			std::uint32_t* i = indexmem.get();
			for (int j = 0; j < num_triangles*3; j++)
			{
				i[j] = indices[j];
			}
		}
	}

	void EyeCandyGeometry::draw() const
	{
		Buffer dummy_mat, dummy_vmat, dummy_tmat;
		app_.setData(in_vertices, out_vertices, dummy_mat, in_indices, dummy_vmat, dummy_tmat, num_verts, 0, num_indices / 3);
		app_.setGlobals();

		renderer->recordDrawingTime(app_.render(0, num_indices / 3));
	}

	void EyeCandyGeometry::draw(int from, int num_indices) const
	{
		Buffer dummy_mat, dummy_vmat, dummy_tmat;
		app_.setData(in_vertices, out_vertices, dummy_mat, in_indices, dummy_vmat, dummy_tmat, num_verts, 0, num_indices / 3);
		app_.setGlobals();

		renderer->recordDrawingTime(app_.render(from, num_indices / 3));
	}
}
