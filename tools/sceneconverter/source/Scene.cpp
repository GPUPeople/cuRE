


#include <tuple>
#include <vector>
#include <set>
#include <unordered_map>

#include <fstream>
#include <iostream>

#include "io.h"

#include "Scene.h"


namespace
{
	class FrameImporter : public virtual SceneBuilder
	{
		std::vector<vertex>& vertices;

		void addVertices(std::vector<vertex>&& v)
		{
			vertices.insert(end(vertices), begin(v), end(v));
		}

		void addSurface(PrimitiveType primitive_type, std::vector<std::uint32_t>&& indices, const char* name, size_t name_length, const char* material_name) {}
		void addMaterial(const char* name, const math::float3& ambient, const math::float3& diffuse, const math::float4& specular, float alpha, const char* tex_name) {}
		void addTexture(const char* name, const char* filename) {}

	public:
		FrameImporter(std::vector<vertex>& vertices)
			: vertices(vertices)
		{
		}
	};

	void buildAdjacency(const surface& surface, std::uint32_t* indices, const std::uint32_t* vertex_coord_map)
	{
		struct halfedge_key_t
		{
			std::uint32_t v0;
			std::uint32_t v1;

			halfedge_key_t(std::uint32_t v0, std::uint32_t v1)
				: v0(v0), v1(v1)
			{
			}

			bool operator ==(const halfedge_key_t& key) const
			{
				return v0 == key.v0 && v1 == key.v1;
			}
		};

		struct halfedge_hash_t
		{
			size_t operator ()(const halfedge_key_t& key) const
			{
				return (key.v0 << 16) ^ key.v1;
			}
		};


		std::unordered_map<halfedge_key_t, std::tuple<size_t, std::uint32_t>, halfedge_hash_t> edge_map;

		for (auto i = surface.start; i < surface.start + surface.num_indices; i += 6)
		{
			for (int j = 0; j < 3; ++j)
			{
				int ii0 = i + ((2 * j) % 6);
				int ii1 = i + ((2 * (j + 1)) % 6);
				int ii2 = i + ((2 * (j + 2)) % 6);

				//if (!edge_map.insert(std::make_pair(halfedge_key_t(vertex_coord_map[indices[ii0]], vertex_coord_map[indices[ii1]]), std::make_tuple(ii0 + 1, indices[ii2]))).second)
				//if (!edge_map.insert(std::make_pair(halfedge_key_t(vertex_coord_map[indices[ii0]], vertex_coord_map[indices[ii1]]), std::make_tuple(ii0 + 1, indices[ii2]))).second)
					//throw std::runtime_error("mesh is non-manifold! (halfedge multiply defined)");

				auto peer = edge_map.find(halfedge_key_t(vertex_coord_map[indices[ii1]], vertex_coord_map[indices[ii0]]));

				if (peer != end(edge_map))
				{
					//auto ip = (peer->second % 6);
					//ip = peer->second - ip + (ip + 4) % 6;
					//indices[ii0 + 1] = indices[ip];
					//indices[peer->second + 1] = indices[ii2];
					indices[ii0 + 1] = std::get<1>(peer->second);
					indices[std::get<0>(peer->second)] = indices[ii2];
				}
			}
		}
	}

}

//void exportScene(const char* filename, const vertex* vertices, size_t num_vertices, const std::uint32_t* indices, size_t num_indices, const surface* surfaces, size_t num_surfaces)
//{
//	std::ofstream file(filename, std::ios::out | std::ios::binary);
//
//	write(file, static_cast<std::uint32_t>(num_vertices));
//	write(file, vertices, num_vertices);
//
//	write(file, static_cast<std::uint32_t>(num_indices));
//	write(file, indices, num_indices);
//
//	write(file, static_cast<std::uint32_t>(num_surfaces));
//
//	for (const surface* surface = surfaces; surface < surfaces + num_surfaces; ++surface)
//	{
//		write(file, surface->start);
//		write(file, surface->num_indices);
//	}
//}


Scene::Scene()
{
}

void Scene::import(import_func_t* import_func, const char* begin, size_t length)
{
	(*import_func)(*this, begin, length);
}

void Scene::importFrame(import_func_t * import_func, const char * begin, size_t length)
{
	FrameImporter importer(vertices);
	(*import_func)(importer, begin, length);
}

void Scene::finalize(bool nomaterial, bool mergeequalmaterials)
{
	/*
	struct vertex_coord_hash_t
	{
	size_t operator ()(const math::float3& p)
	{
	return (*reinterpret_cast<const std::uint32_t*>(&p.x) << 16U) ^
	(*reinterpret_cast<const std::uint32_t*>(&p.y) <<	8U) ^
	(*reinterpret_cast<const std::uint32_t*>(&p.z));
	}
	};

	std::uint32_t coordinates = 0U;
	std::vector<std::uint32_t> vertex_coord_map(vertices.size());
	std::unordered_map<math::float3, std::uint32_t, vertex_coord_hash_t> coord_vertex_map;

	for (int i = 0; i < vertices.size(); ++i)
	{
	auto found = coord_vertex_map.find(vertices[i].p);
	if (found != end(coord_vertex_map))
	{
	vertex_coord_map[i] = found->second;
	}
	else
	{
	coord_vertex_map.insert(std::make_pair(vertices[i].p, coordinates));
	vertex_coord_map[i] = coordinates++;
	}
	}

	for (const auto& surface : surfaces)
	buildAdjacency(surface, &indices[0], &vertex_coord_map[0]);*/

	if (nomaterial)
	{ 
		for (auto & surface : surfaces)
			surface.matId = -1;
		textures.clear();
		materials.clear();
	}
	if (mergeequalmaterials)
	{
		decltype(surfaces) newsurfaces;
		decltype(indices) newindices;
		std::set<int> donemats;
		for (int i = 0; i < surfaces.size(); ++i)
		{
			if (donemats.find(surfaces[i].matId) != donemats.end())
				continue;
			donemats.insert(surfaces[i].matId);
			bool first = true;
			for (int j = i; j < surfaces.size(); ++j)
			{
				if (surfaces[i].matId == surfaces[j].matId)
				{
					if (first)
					{
						newsurfaces.push_back(surface(surfaces[i].name.c_str(), surfaces[i].name.size(), surfaces[i].primitive_type, static_cast<std::uint32_t>(newindices.size()), 0, surfaces[i].matId));
					}
					first = false;
					newindices.insert(end(newindices), indices.begin() + surfaces[j].start, indices.begin() + surfaces[j].start + surfaces[j].num_indices);
					newsurfaces.back().num_indices += surfaces[j].num_indices;
				}
			}
		}
		newsurfaces.swap(surfaces);
		newindices.swap(indices);
	}
	//remove unreferenced materials
	{
		std::set<int> existingmaterials;
		for (const auto & surf : surfaces)
			existingmaterials.insert(surf.matId);

		decltype(materials) newmaterials;
		std::map<int, int> matmapping;
		for (int i = 0; i < materials.size(); ++i)
		{
			if (existingmaterials.find(i) != existingmaterials.end())
			{
				matmapping.insert(std::make_pair(i, static_cast<int>(newmaterials.size())));
				newmaterials.push_back(materials[i]);
			}
		}
		if (materials.size() != newmaterials.size())
		{ 
			std::cout << "Removed " << (materials.size() - newmaterials.size()) << " unreferenced materials\n";
			newmaterials.swap(materials);
		
			for (auto & surf : surfaces)
			{
				auto f = matmapping.find(surf.matId);
				if (f != matmapping.end())
					surf.matId = f->second;
			}
		}
	}
	//remove unreferenced textures
	{
		std::set<int> existingtextures;
		for (const auto & mat : materials)
			existingtextures.insert(mat.texId);

		decltype(textures) newtextures;
		std::map<int, int> texmapping;
		for (int i = 0; i < textures.size(); ++i)
		{
			if (existingtextures.find(i) != existingtextures.end())
			{
				texmapping.insert(std::make_pair(i, static_cast<int>(newtextures.size())));
				newtextures.push_back(textures[i]);
			}
		}
		if (textures.size() != newtextures.size())
		{
			std::cout << "Removed " << (textures.size() - newtextures.size()) << " unreferenced materials\n";
			newtextures.swap(textures);
			for (auto & mat : materials)
			{
				auto f = texmapping.find(mat.texId);
				if (f != texmapping.end())
					mat.texId = f->second;
			}
		}
	}
}

void Scene::addVertices(std::vector<vertex>&& v)
{
	vertices.insert(end(vertices), begin(v), end(v));
}

void Scene::addSurface(PrimitiveType primitive_type, std::vector<std::uint32_t>&& ind, const char* name, size_t name_length, const char* mat_name)
{
	//if (ind.size() % 3 != 0)
	//	throw std::runtime_error("surface is not a triangle mesh!");

	int mat = -1;
	if (mat_name != nullptr)
	{
		auto found = materialMapping.find(mat_name);
		if (found == materialMapping.end())
		{
			//if (!ignorematerrors)
			//	throw std::runtime_error((std::string("material with the name ") + mat_name + " not found").c_str());
		}
		else
			mat = found->second;
	}


	surfaces.emplace_back(name, name_length, primitive_type, static_cast<std::uint32_t>(indices.size()), static_cast<std::uint32_t>(ind.size()), mat);
	size_t start = indices.size();
	//indices.resize(indices.size() + ind.size());
	indices.insert(end(indices), begin(ind), end(ind));

	//auto src = begin(ind);
	//auto dest = begin(indices) + start;

	//while (dest < end(indices))
	//{
	//	auto i0 = *src++;
	//	auto i1 = *src++;
	//	auto i2 = *src++;

	//	*dest++ = i0;
	//	*dest++ = i2;
	//	*dest++ = i1;
	//}

	//auto dest = begin(indices) + start;
	//
	//for (auto i : ind)
	//{
	//	*dest++ = i;
	//	*dest++ = i;
	//}
}

bool Scene::hasTexture(const char* name)
{
	return textureMapping.find(name) != textureMapping.end();
}

bool Scene::hasMaterial(const char* name)
{
	return materialMapping.find(name) != materialMapping.end();
}

void Scene::addMaterial(const char* name, const math::float3& ambient, const math::float3& diffuse, const math::float4& specular, float alpha, const char* tex_name)
{
	if (materialMapping.find(name) == materialMapping.end())
	{
		int tex = -1;
		if (tex_name != nullptr)
		{
			auto found = textureMapping.find(tex_name);
			if (found == textureMapping.end())
			{
				//if (!ignorematerrors)
				//	throw std::runtime_error((std::string("texture with the name ") + tex_name + " not found").c_str());
			}
			else
			tex = found->second;
		}
		materials.emplace_back(name, ambient, diffuse, specular, alpha, tex);
		materialMapping.insert(std::make_pair(name, static_cast<int>(materials.size()) - 1));
	}
}
void Scene::addTexture(const char* name, const char* filename)
{
	if (textureMapping.find(name) == textureMapping.end())
	{ 
		textures.emplace_back(name, filename);
		textureMapping.insert(std::make_pair(name, static_cast<int>(textures.size()) - 1));
	}
}

void Scene::serialize(const char* filename, export_func_t* export_func, bool nomaterial, bool mergeequalmaterials)
{
	finalize(nomaterial, mergeequalmaterials);

	std::cout << "writing scene file \"" << filename << "\"\n";
	std::cout << "\t" << vertices.size() << " vertices\n";
	std::cout << "\t" << indices.size() << " indices\n";
	std::cout << "\t" << surfaces.size() << " surfaces\n";
	std::cout << "\t" << materials.size() << " materials\n";
	std::cout << "\t" << textures.size() << " textures\n";

	std::ofstream file(filename, std::ios::out | std::ios::binary);
	export_func(file, &vertices[0], vertices.size(), &indices[0], indices.size(), &surfaces[0], surfaces.size(), materials.empty() ? nullptr : &materials[0], materials.size(), textures.empty() ? nullptr : &textures[0], textures.size());
}
