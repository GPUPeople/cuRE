


#ifndef INCLUDED_LOADEDSCENE
#define INCLUDED_LOADEDSCENE

#pragma once

#include <tuple>
#include <vector>
#include <map>

#include <math/vector.h>
#include <math/matrix.h>

#include <image.h>
#include <rgba8.h>

#include <SceneBuilder.h>

#include "Resource.h"
#include "Renderer.h"
#include "Camera.h"
#include "Scene.h"

#include "resource_ptr.h"


class LoadedScene : public Scene, private SceneBuilder
{
	std::vector<std::tuple<size_t, size_t>> scene_geom_mat_pairings;
	std::vector<resource_ptr<Geometry>> scene_geometries;
	std::vector<resource_ptr<Texture>> scene_textures;
	std::vector<resource_ptr<Material>> scene_materials;

	void addVertices(std::vector<vertex>&& vertices);
	void addSurface(PrimitiveType primitive_type, std::vector<std::uint32_t>&& indices, const char* name, size_t name_length, const char* material_name);
	void addMaterial(const char* name, const math::float3& ambient, const math::float3& diffuse, const math::float4& specular, float alpha, const char* tex_name);
	void addTexture(const char* name, const char* filename);
	bool hasTexture(const char* name);
	bool hasMaterial(const char* name);

	std::vector<float> vertices;
	std::vector<float> normals;
	std::vector<float> texcoords;
	std::vector<std::uint32_t> indices;

	std::vector<surface> surfaces;
	std::vector<material> materials;
	std::vector<std::tuple<std::unique_ptr<std::uint32_t[]>, size_t, size_t, int>> textures;

	std::map<std::string, int> texture_mapping;
	std::map<std::string, int> material_mapping;

public:
	LoadedScene(const char* scenefile);

	LoadedScene(const LoadedScene&) = delete;
	LoadedScene& operator =(const LoadedScene&) = delete;

	void switchRenderer(Renderer* renderer);

	void draw(RendereringContext* context) const;

	void save(Config& config) const override {}
};


class HeavyScene : public Scene, private SceneBuilder
{
public:
	enum class Type
	{
		VERTEX_HEAVY,
		FRAGMENT_HEAVY
	};

private:
	resource_ptr<Material> material;
	resource_ptr<Geometry> geometry;

	void addVertices(std::vector<vertex>&& vertices);
	void addSurface(PrimitiveType primitive_type, std::vector<std::uint32_t>&& indices, const char* name, size_t name_length, const char* material_name);
	void addMaterial(const char* name, const math::float3& ambient, const math::float3& diffuse, const math::float4& specular, float alpha, const char* tex_name);
	void addTexture(const char* name, const char* filename);

	std::vector<float> vertices;
	std::vector<float> normals;
	std::vector<float> texcoords;
	std::vector<std::uint32_t> indices;

	Type type;

public:
	HeavyScene(const char* scenefile, Type type);

	HeavyScene(const LoadedScene&) = delete;
	HeavyScene& operator =(const HeavyScene&) = delete;

	void switchRenderer(Renderer* renderer);

	void draw(RendereringContext* context) const;

	void save(Config& config) const override {}
};

#endif  // INCLUDED_SCENE
