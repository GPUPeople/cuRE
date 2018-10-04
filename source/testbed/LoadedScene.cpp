


#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <type_traits>

#include <vector>
#include <map>
#include <set>

#include <math/vector.h>

#include "LoadedScene.h"
#include <binscene.h>
#include <png.h>


//TODO: this is rly ugly....
#include <binscene.cpp>


namespace
{
	bool fileExists(const char* filename)
	{
		std::ifstream file(filename);
		return !!file;
	}

	std::string textureLayerName(const std::string& basename, int i)
	{
		std::ostringstream fn;
		fn << basename << "." << i << ".png";
		return fn.str();
	}

	template <typename InputIt, typename OutputIt>
	OutputIt flipCopy(OutputIt dest, const InputIt src, size_t w, size_t h)
	{
		for (int r = 0; r < h; ++r)
		{
			auto s = src + (h - 1 - r) * w;

			std::copy(s, s + w, dest);
			dest += w;
		}

		return dest;
	}
}


LoadedScene::LoadedScene(const char* scenefile)
{
	std::ifstream in(scenefile, std::ios::binary);

	if (!in)
		throw std::runtime_error("unable to open file " + std::string(scenefile));

	in.seekg(0, std::ios::end);

	size_t file_size = in.tellg();
	std::unique_ptr<char[]> buffer(new char[file_size]);

	in.seekg(0, std::ios::beg);
	in.read(&buffer[0], file_size);

	binscene::read(*this, &buffer[0], file_size);

	// add default mat
	math::float3 amb(0, 0, 0), diff(0.8, 0.8, 0.8);
	math::float4 spec(0.1, 0.1, 0.1, 1.0);
	LoadedScene::addMaterial("DummyDef010", amb, diff, spec, 1.0f, nullptr);

	std::cout << "Loaded scene has\n"
		<< "\t" << vertices.size() << " vertices\n"
		<< "\t" << indices.size() << " indices\n"
		<< "\t" << surfaces.size() << " surfaces\n"
		<< "\t" << materials.size() << " materials\n"
		<< "\t" << textures.size() << " textures\n\n";
}


void LoadedScene::addVertices(std::vector<vertex>&& v)
{
	vertices.resize(vertices.size() + 3 * v.size());
	normals.resize(normals.size() + 3 * v.size());
	texcoords.resize(texcoords.size() + 2 * v.size());

	float *p = &vertices.back() - 3 * v.size() + 1,
	      *n = &normals.back() - 3 * v.size() + 1,
	      *t = &texcoords.back() - 2 * v.size() + 1;
	
	for (auto &vertex : v)
	{
		*p++ = vertex.p.x; *p++ = vertex.p.y; *p++ = vertex.p.z;
		*n++ = vertex.n.x; *n++ = vertex.n.y; *n++ = vertex.n.z;
		*t++ = vertex.t.x; *t++ = vertex.t.y;
	}
}

void LoadedScene::addSurface(PrimitiveType primitive_type, std::vector<std::uint32_t>&& ind, const char* name, size_t name_length, const char* mat_name)
{
	//if (ind.size() % 3 != 0)
	//	throw std::runtime_error("surface is not a triangle mesh!");

	int mat = -1;
	if (mat_name != nullptr)
	{
		auto found = material_mapping.find(mat_name);
		if (found == material_mapping.end())
			throw std::runtime_error((std::string("material with the name ") + mat_name + " not found").c_str());
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

}

bool LoadedScene::hasTexture(const char* name)
{
	return texture_mapping.find(name) != texture_mapping.end();
}

bool LoadedScene::hasMaterial(const char* name)
{
	return material_mapping.find(name) != material_mapping.end();
}

void LoadedScene::addMaterial(const char* name, const math::float3& ambient, const math::float3& diffuse, const math::float4& specular, float alpha, const char* tex_name)
{
	if (material_mapping.find(name) == material_mapping.end())
	{
		int tex = -1;
		if (tex_name != nullptr)
		{
			auto found = texture_mapping.find(tex_name);
			if (found == texture_mapping.end())
				throw std::runtime_error((std::string("texture with the name ") + tex_name + " not found").c_str());
			tex = found->second;
		}
		materials.emplace_back(name, ambient, diffuse, specular, alpha, tex);
		material_mapping.insert(std::pair<std::string, int>(name, static_cast<int>(materials.size()) - 1));
	}
}

void LoadedScene::addTexture(const char* name, const char* filename)
{
	if (texture_mapping.find(name) == texture_mapping.end())
	{
		std::unique_ptr<std::uint32_t[]> texdata;
		unsigned int texw = 256U;
		unsigned int texh = 256U;

		std::string basename = std::string("assets/") + filename;

		auto layer_image_0 = textureLayerName(basename, 0);

		if (fileExists(layer_image_0.c_str()))
		{
			auto size = PNG::readSize(layer_image_0.c_str());
			texw = static_cast<unsigned int>(std::get<0>(size));
			texh = static_cast<unsigned int>(std::get<1>(size));
		}

		texdata = std::make_unique<std::uint32_t[]>(texw * texh * 4);


		int texlevels = 0;
		auto teximage = &texdata[0];

		static const int pattern_size = 6;
		
		for (unsigned int w = texw, h = texh; w > 1 || h > 1; w = std::max(w / 2U, 1U), h = std::max(h / 2U, 1U))
		{
			auto layer_image = textureLayerName(basename, texlevels);

			if (fileExists(layer_image.c_str()))
			{
				auto png = PNG::loadRGBA8(layer_image.c_str());

				if (width(png) != w || height(png) != h)
					throw std::runtime_error("mip level image dimension mismatch");

				teximage = flipCopy(teximage, data(png), w, h);
			}
			else
			{
				const int s = std::max(pattern_size - texlevels, 0);

				for (unsigned int y = 0; y < h; ++y)
				{
					for (unsigned int x = 0; x < w; ++x)
					{
						*teximage++ = ((y >> s) & 0x1U) ^ ((x >> s) & 0x1U) ? 0xFFFFFFFFU : 0xFF000000U;
					}
				}
			}

			++texlevels;
			//break;
		}

		textures.emplace_back(std::move(texdata), texw, texh, texlevels);
		texture_mapping.insert(std::make_pair(name, static_cast<int>(textures.size()) - 1));
	}
}

void LoadedScene::switchRenderer(Renderer* renderer)
{
	if (renderer)
	{
		// create textures
		scene_textures.reserve(textures.size());
		for (auto& tex : textures)
			scene_textures.emplace_back(std::move(renderer->createTexture2DRGBA8(std::get<1>(tex), std::get<2>(tex), std::get<3>(tex), &std::get<0>(tex)[0])));

		scene_materials.reserve(materials.size());
		for (auto& mat : materials)
		{
			if (mat.texId != -1 && scene_textures[mat.texId] != nullptr)
				scene_materials.emplace_back(std::move(scene_textures[mat.texId]->createTexturedLitMaterial(math::float4(mat.diffuse, mat.alpha))));
			else
				scene_materials.emplace_back(std::move(renderer->createLitMaterial(math::float4(mat.diffuse, mat.alpha))));
		}

		scene_geometries.reserve(surfaces.size());
		for (auto& surf : surfaces)
		{
			scene_geom_mat_pairings.push_back(std::tuple<size_t, size_t>(scene_geometries.size(), surf.matId));
			if (surf.primitive_type == PrimitiveType::QUADS)
				scene_geometries.emplace_back(std::move(renderer->createIndexedQuads(&vertices[0], &normals[0], &texcoords[0], vertices.size() / 3, &indices[surf.start], surf.num_indices)));
			else
				scene_geometries.emplace_back(std::move(renderer->createIndexedTriangles(&vertices[0], &normals[0], &texcoords[0], vertices.size() / 3, &indices[surf.start], surf.num_indices)));
		}

		if (std::any_of(begin(scene_textures), end(scene_textures), [](auto&& p) { return !p; }) ||
			 std::any_of(begin(scene_geometries), end(scene_geometries), [](auto&& p) { return !p; }) ||
			 std::any_of(begin(scene_materials), end(scene_materials), [](auto&& p) { return !p; }))
			throw std::runtime_error("renderer cannot support this scene type");
	}
	else
	{
		scene_geom_mat_pairings.clear();
		scene_geometries.clear();
		scene_textures.clear();
		scene_materials.clear();
	}
}

void LoadedScene::draw(RendereringContext* context) const
{
	context->setLight(math::float3(0.0f, 10.0f, 0.0f), math::float3(1.0f, 1.0f, 1.0f));

	context->setObjectTransform(math::float3x4(1.0f, 0.0f, 0.0f, 0.0f,
	                                           0.0f, 1.0f, 0.0f, 0.0f,
	                                           0.0f, 0.0f, 1.0f, 0.0f));

	for (auto &todraw : scene_geom_mat_pairings)
	{
		int mat = static_cast<int>(std::get<1>(todraw));
		if (mat == -1)
			mat = static_cast<int>(scene_materials.size()) - 1;
		scene_materials[mat]->draw(scene_geometries[std::get<0>(todraw)].get());
	}
}



void HeavyScene::addVertices(std::vector<vertex>&& v)
{
	vertices.resize(vertices.size() + 3 * v.size());
	normals.resize(normals.size() + 3 * v.size());
	texcoords.resize(texcoords.size() + 2 * v.size());

	float *p = &vertices.back() - 3 * v.size() + 1, *n = &normals.back() - 3 * v.size() + 1, *t = &texcoords.back() - 2 * v.size() + 1;

	for (const auto& vertex : v)
	{
		*p++ = vertex.p.x; *p++ = vertex.p.y; *p++ = vertex.p.z;
		*n++ = vertex.n.x; *n++ = vertex.n.y; *n++ = vertex.n.z;
		*t++ = vertex.t.x; *t++ = vertex.t.y;
	}
}

void HeavyScene::addSurface(PrimitiveType primitive_type, std::vector<std::uint32_t>&& ind, const char* name, size_t name_length, const char* material_name)
{
	indices.insert(end(indices), begin(ind), end(ind));
}

void HeavyScene::addMaterial(const char* name, const math::float3& ambient, const math::float3& diffuse, const math::float4& specular, float alpha, const char* tex_name)
{
}

void HeavyScene::addTexture(const char* name, const char* filename)
{
}

HeavyScene::HeavyScene(const char* scenefile, Type type)
	: type(type)
{
	std::ifstream in(scenefile, std::ios::binary);

	if (!in)
		throw std::runtime_error("unable to open file " + std::string(scenefile));

	in.seekg(0, std::ios::end);

	size_t file_size = in.tellg();
	std::unique_ptr<char[]> buffer(new char[file_size]);

	in.seekg(0, std::ios::beg);
	in.read(&buffer[0], file_size);

	binscene::read(*this, &buffer[0], file_size);

	std::cout << "heavy scene has\n"
		<< "\t" << vertices.size() << " vertices\n"
		<< "\t" << indices.size() << " indices\n";
}

void HeavyScene::switchRenderer(Renderer* renderer)
{
	if (renderer)
	{
		switch (type)
		{
		case Type::VERTEX_HEAVY:
			material.reset(renderer->createVertexHeavyMaterial(16));
			break;

		case Type::FRAGMENT_HEAVY:
			material.reset(renderer->createFragmentHeavyMaterial(16));
			break;
		}

		geometry.reset(renderer->createIndexedTriangles(data(vertices), data(normals), data(texcoords), size(vertices)/3, data(indices), size(indices)));
	}
	else
	{
		material.reset();
		geometry.reset();
	}
}

void HeavyScene::draw(RendereringContext* context) const
{
	//context->setLight(math::float3(0.0f, 10.0f, 0.0f), math::float3(1.0f, 1.0f, 1.0f));

	context->setObjectTransform(math::float3x4(1.0f, 0.0f, 0.0f, 0.0f,
	                                           0.0f, 1.0f, 0.0f, 0.0f,
	                                           0.0f, 0.0f, 1.0f, 0.0f));

	material->draw(geometry.get());
}
