


#include <cstdint>
#include <type_traits>
#include <iostream>

#include <vector>
#include <map>
#include <set>

#include <math/vector.h>

#include "SphereScene.h"

using math::float2;
using math::float3;


namespace
{
	const float phi = 1.6180339887498948482045868343656f;
	const float iphi = 1.0f / phi;

	namespace Tetrahedron
	{
		const float3 positions[] = {
			float3(-1.0f,  0.0f, -1.0f / sqrtf(2.0f)),
			float3( 1.0f,  0.0f, -1.0f / sqrtf(2.0f)),
			float3( 0.0f, -1.0f,  1.0f / sqrtf(2.0f)),
			float3( 0.0f,  1.0f,  1.0f / sqrtf(2.0f))
		};
		const std::uint32_t indices[] = {
			1, 0, 3,
			3, 0, 2,
			2, 0, 1,
			3, 2, 1
		};
		size_t num_vertices = std::extent<decltype(positions)>::value;
		size_t num_indices = std::extent<decltype(indices)>::value;
	}
	namespace Ikosmooth
	{
		const float t = (1.0f + sqrtf(5.0f)) / 2.0f;
		const float3 positions[] = {
			float3(-1.0f,    t,  0.0f),
			float3( 1.0f,    t,  0.0f),
			float3(-1.0f,   -t,  0.0f),
			float3( 1.0f,   -t,  0.0f),
			float3( 0.0f, -1.0f,    t),
			float3( 0.0f,  1.0f,    t),
			float3( 0.0f, -1.0f,   -t),
			float3( 0.0f,  1.0f,   -t),
			float3(    t, 0.0f, -1.0f),
			float3(    t, 0.0f,  1.0f),
			float3(   -t, 0.0f, -1.0f),
			float3(   -t, 0.0f,  1.0f)
		};
		const std::uint32_t indices[] = {
			 0, 11,  5,
			 0,  5,  1,
			 0,  1,  7,
			 0,  7, 10,
			 0, 10, 11,

			 1,  5,  9,
			 5, 11,  4,
			11, 10,  2,
			10,  7,  6,
			 7,  1,  8,

			 3,  9,  4,
			 3,  4,  2,
			 3,  2,  6,
			 3,  6,  8,
			 3,  8,  9,

			 4,  9,  5,
			 2,  4, 11,
			 6,  2, 10,
			 8,  6,  7,
			 9,  8,  1
		};

		size_t num_vertices = std::extent<decltype(positions)>::value;
		size_t num_indices = std::extent<decltype(indices)>::value;
	}

	class SphereMesh
	{
		std::map<std::uint32_t, std::set<std::uint32_t> > neighbors;
		std::vector<float3> _vertices;
		std::vector<std::uint32_t> _indices;
		std::set<std::uint32_t> _marks;

		void addNeighbor(std::uint32_t a, std::uint32_t b)
		{
			auto found = neighbors.find(a);
			if (found == neighbors.end())
				neighbors.insert(std::make_pair(a, std::set<std::uint32_t>(&b, &b + 1)));
			else
				found->second.insert(b);
		}
		void insertNeighbors(std::uint32_t a, std::uint32_t b)
		{
			addNeighbor(a, b);
			addNeighbor(b, a);
		}

		template<class It>
		static bool contains(It a, It b, std::uint32_t i0, std::uint32_t i1, std::uint32_t i2, std::uint32_t& f)
		{
			for (; a != b; ++a)
			{
				if (*a == i0 || *a == i1 || *a == i2)
				{
					f = *a;
					return true;
				}
			}
			return false;
		}
		static std::uint32_t remapId(std::uint32_t origId, const std::vector<std::uint32_t>& removeIDs)
		{
			for (std::uint32_t i = 0; i < removeIDs.size(); ++i)
			{
				if (removeIDs[i] >= origId)
					return origId - i;
			}
			return origId - static_cast<std::uint32_t>(removeIDs.size());
		}
	public:

		std::vector<float3>& positions() { return _vertices; }
		std::vector<std::uint32_t>& indices() { return _indices; }

		std::uint32_t addVertex(const float3& v)
		{
			_vertices.push_back(normalize(v));
			return static_cast<std::uint32_t>(_vertices.size() - 1);
		}
		void addTriangle(std::uint32_t i0, std::uint32_t i1, std::uint32_t i2)
		{
			_indices.push_back(i0); _indices.push_back(i1); _indices.push_back(i2);
			insertNeighbors(i0, i1);
			insertNeighbors(i0, i2);
			insertNeighbors(i1, i2);
		}
		void markVertices()
		{
			_marks = decltype(_marks)(_indices.begin(), _indices.end());
		}
		void subdivide()
		{
			neighbors.clear();
			decltype(_vertices) temp_vertices;
			decltype(_indices) temp_indices;
			decltype(_marks) temp_marks;
			_vertices.swap(temp_vertices);
			_indices.swap(temp_indices);
			_marks.swap(temp_marks);

			_vertices.reserve(5 * temp_vertices.size());
			_indices.reserve(8 * temp_indices.size());

			std::map<std::uint32_t, std::uint32_t> alreadyinserted_positions;
			std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> alreadygenerated_positions;
			for (size_t i = 0; i < temp_indices.size(); i += 3)
			{
				std::uint32_t subdiv_ids[6];
				for (size_t edge = 0; edge < 3; ++edge)
				{
					std::uint32_t v0id = temp_indices[i + edge],
					              v1id = temp_indices[i + (edge + 1) % 3];
					auto found = alreadyinserted_positions.find(v0id);
					if (found == alreadyinserted_positions.end())
					{
						_vertices.push_back(temp_vertices[v0id]);
						std::uint32_t tid = subdiv_ids[2 * edge] = static_cast<std::uint32_t>(_vertices.size() - 1);
						alreadyinserted_positions.insert(std::make_pair(v0id, tid));
						if (temp_marks.find(v0id) != temp_marks.end())
							_marks.insert(tid);
					}
					else
						subdiv_ids[2 * edge] = found->second;

					auto searchedge = std::make_pair(std::min(v0id, v1id), std::max(v0id, v1id));
					auto foundedge = alreadygenerated_positions.find(searchedge);
					if (foundedge == alreadygenerated_positions.end())
					{
						float3 p = normalize(temp_vertices[v0id] + temp_vertices[v1id]);
						_vertices.push_back(p);
						std::uint32_t tid = subdiv_ids[2 * edge + 1] = static_cast<std::uint32_t>(_vertices.size() - 1);
						alreadygenerated_positions.insert(std::make_pair(searchedge, tid));
					}
					else
						subdiv_ids[2 * edge + 1] = foundedge->second;
				}
				addTriangle(subdiv_ids[0], subdiv_ids[1], subdiv_ids[5]);
				addTriangle(subdiv_ids[5], subdiv_ids[1], subdiv_ids[3]);
				addTriangle(subdiv_ids[1], subdiv_ids[2], subdiv_ids[3]);
				addTriangle(subdiv_ids[5], subdiv_ids[3], subdiv_ids[4]);
			}
		}
		void optimize(float strength)
		{
			//0.0f < strength
			if (strength < 0)
				return;

			// optimize all vertex positions
			decltype(_vertices) temp_vertices(_vertices);
			for (std::uint32_t i = 0; i < _vertices.size(); ++i)
			{
				auto ninfo = neighbors.find(i);
				if (ninfo == neighbors.end())
					std::cout << "Warning: vertex with id " << i << " is not referenced in mesh, not going to optimize it\n";
				else
				{
					/*
					// get original vertex
					float3 v = temp_vertices[i] * (1.0f - strength);
					float nweight = strength/static_cast<float>(ninfo->second.size());
					for (auto nit : ninfo->second)
					v += nweight*temp_vertices[nit];
					_vertices[i] = normalize(v);
					*/
					float3 v = temp_vertices[i];
					float3 offset(0, 0, 0);
					float avglength = 0;
					for (auto nit : ninfo->second)
						avglength += length2(v - temp_vertices[nit]);
					avglength /= static_cast<float>(ninfo->second.size());
					for (auto nit : ninfo->second)
					{
						float3 d = temp_vertices[nit] - v;
						offset += (length2(d) - avglength) * d;
					}
					_vertices[i] = normalize(v + strength*offset);
				}
			}
		}
		void tryRemoveMarked()
		{
			std::vector<std::uint32_t> canremove;
			for (auto it : _marks)
			{
				auto found = neighbors.find(it);
				if (found == neighbors.end() || found->second.size() == 3)
					canremove.push_back(it);
			}
			if (canremove.size() > 0)
			{
				// id remap function
				std::sort(canremove.begin(), canremove.end());

				// remove triangles
				neighbors.clear();
				decltype(_vertices) temp_vertices;
				decltype(_indices) temp_indices;
				decltype(_marks) temp_marks;
				_vertices.swap(temp_vertices);
				_indices.swap(temp_indices);
				_marks.swap(temp_marks);

				std::map<std::uint32_t, std::vector<std::uint32_t> > toGenerate;

				for (size_t i = 0; i < temp_indices.size(); i += 3)
				{
					std::uint32_t res;
					if (!contains(canremove.begin(), canremove.end(), temp_indices[i], temp_indices[i + 1], temp_indices[i + 2], res))
					{
						addTriangle(remapId(temp_indices[i], canremove), remapId(temp_indices[i + 1], canremove), remapId(temp_indices[i + 2], canremove));
						for (size_t j = 0; j < 3; ++j)
							if (temp_marks.find(temp_indices[i + j]) != temp_marks.end())
								_marks.insert(remapId(temp_indices[i + j], canremove));
					}
					else
					{
						auto found = toGenerate.find(res);
						if (found == toGenerate.end())
						{
							found = toGenerate.insert(std::make_pair(res, std::vector<std::uint32_t>())).first;
							if (temp_indices[i] == res)
								found->second.push_back(temp_indices[i + 1]),
								found->second.push_back(temp_indices[i + 2]);
							else if (temp_indices[i + 2] == res)
								found->second.push_back(temp_indices[i + 0]),
								found->second.push_back(temp_indices[i + 1]);
							else
								found->second.push_back(temp_indices[i + 2]),
								found->second.push_back(temp_indices[i + 0]);
						}
						else if (found->second.size() == 2)
							for (size_t j = 0; j < 3; ++j)
								if (temp_indices[i + j] != res && found->second[0] != temp_indices[i + j] && found->second[1] != temp_indices[i + j])
									found->second.push_back(temp_indices[i + j]);
					}

				}
				for (auto id : canremove)
				{
					// add new triangles
					for (auto it : toGenerate)
						addTriangle(remapId(it.second[0], canremove), remapId(it.second[1], canremove), remapId(it.second[2], canremove));
				}

				//reenter remaining vertices
				canremove.push_back(static_cast<std::uint32_t>(temp_vertices.size()));
				_vertices.reserve(temp_vertices.size());
				for (size_t i = 0; i + 1 < canremove.size(); ++i)
					_vertices.insert(_vertices.end(), temp_vertices.begin() + canremove[i] + 1, temp_vertices.begin() + canremove[i + 1]);
			}
		}
	};
}

SphereScene::SphereScene()
{
}

void SphereScene::switchRenderer(Renderer* renderer)
{
	if (renderer)
	{
		material = resource_ptr<Material>(renderer->createLitMaterial(math::float4(1.0f, 1.0f, 1.0f, 1.0f)));
		
		SphereMesh mesh;
		{
			//using namespace Tetrahedron;
			using namespace Ikosmooth;
			for (auto& pos : positions)
				mesh.addVertex(pos);
			for (size_t i = 0; i < num_indices; i += 3)
				mesh.addTriangle(indices[i], indices[i + 1], indices[i + 2]);
			mesh.markVertices();
		}



		size_t refinements = 6;
		for (size_t ref = 0; ref < refinements; ++ref)
		{
			mesh.subdivide();
			if (ref == 0)
				mesh.tryRemoveMarked();
			if (ref + 3 >= refinements)
				for (int i = 0; i < 3*ref; ++i)
					mesh.optimize(1.0f);
		}
				
		geometry = resource_ptr<Geometry>(renderer->createIndexedTriangles(&mesh.positions()[0].x, &mesh.positions()[0].x, &mesh.positions()[0].x, mesh.positions().size(), &mesh.indices()[0], mesh.indices().size()));
		std::cout << "generated sphere has " << mesh.positions().size() << " vertices and " << mesh.indices().size() << " indices.\n";
	}
	else
	{
		geometry.reset();
		material.reset();
	}
}

void SphereScene::draw(RendereringContext* context) const
{
	context->setLight(math::float3(0.0f, 10.0f, 0.0f), math::float3(1.0f, 1.0f, 1.0f));

	float gridoffset = 0.f;
	float gridstep = 2.f;
	for (float z = -gridoffset; z <= gridoffset + 0.001f; z += gridstep)
		for (float x = -gridoffset; x <= gridoffset + 0.001f; x += gridstep)
		{ 
			context->setObjectTransform(math::float3x4(1.0f, 0.0f, 0.0f, x,
			                                           0.0f, 1.0f, 0.0f, 0.0f,
			                                           0.0f, 0.0f, 1.0f, z));
			material->draw(geometry.get());
		}

}
