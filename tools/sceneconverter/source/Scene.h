


#ifndef INCLUDED_SCENE
#define INCLUDED_SCENE

#pragma once

#include <cstdint>
#include <vector>
#include <map>

#include <iosfwd>

#include "SceneBuilder.h"


typedef void import_func_t(SceneBuilder& builder, const char* begin, size_t length);
typedef void export_func_t(std::ostream& file, const vertex* vertices, size_t num_vertices, const std::uint32_t* indices, size_t num_indices, const surface* surfaces, size_t num_surfaces, const material* materials, size_t num_materials, const texture* textures, size_t num_textures);

class Scene : private virtual SceneBuilder
{
	std::vector<vertex> vertices;
	std::vector<std::uint32_t> indices;

	std::vector<surface> surfaces;
	std::vector<material> materials;
	std::vector<texture> textures;

	std::map<std::string, int> materialMapping;
	std::map<std::string, int> textureMapping;

	void addVertices(std::vector<vertex>&& vertices);
	void addSurface(PrimitiveType primitive_type, std::vector<std::uint32_t>&& indices, const char* name, size_t name_length, const char* material_name);
	void addMaterial(const char* name, const math::float3& ambient, const math::float3& diffuse, const math::float4& specular, float alpha, const char* tex_name);
	void addTexture(const char* name, const char* filename);
	bool hasTexture(const char* name);
	bool hasMaterial(const char* name);

	void finalize(bool nomaterial, bool mergeequalmaterials);

public:
	Scene();

	Scene(const Scene&) = delete;
	Scene& operator =(const Scene&) = delete;
	
	void import(import_func_t* import_func, const char* begin, size_t length);
	void importFrame(import_func_t* import_func, const char* begin, size_t length);

	void serialize(const char* filename, export_func_t* export_func, bool nomaterial = false, bool mergeequalmaterials = false);
};

void exportScene(const char* filename, const vertex* vertices, size_t num_vertices, const std::uint32_t* indices, size_t num_indices, const surface* surfaces, size_t num_surfaces);

#endif  // INCLUDED_SCENE
