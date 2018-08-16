


#ifndef INCLUDED_SCENE_BUILDER
#define INCLUDED_SCENE_BUILDER

#pragma once

#include <cstdint>
#include <vector>

#include <math/vector.h>


struct vertex
{
	math::float3 p;
	math::float3 n;
	math::float2 t;

	vertex() = default;

	vertex(const math::float3& p, const math::float3& n, const math::float2& t)
		: p(p), n(n), t(t)
	{
	}
};

struct texture
{
	std::string name;
	std::string fname;

	texture() = default;

	texture(const char* name, const char* fname)
		: name(name), fname(fname)
	{
	}
};

struct material
{
	std::string name;
	math::float3 ambient;
	math::float3 diffuse;
	math::float4 specular;
	float alpha;
	int texId;

	material() = default;

	material(const char* name, const math::float3& ambient, const math::float3& diffuse, const math::float4& specular, float alpha, int texId = -1)
		: name(name), ambient(ambient), diffuse(diffuse), specular(specular), alpha(alpha), texId(texId)
	{
	}
};

enum class PrimitiveType
{
	TRIANGLES,
	QUADS
};

struct surface
{
	std::string name;
	PrimitiveType primitive_type;
	std::uint32_t start;
	std::uint32_t num_indices;
	int matId;

	surface() = default;

	surface(const char* name, size_t name_length, PrimitiveType primitive_type, std::uint32_t start, std::uint32_t num_indices, int matId = -1)
		: name(name, name_length), primitive_type(primitive_type), start(start), num_indices(num_indices), matId(matId)
	{
	}
};

class SceneBuilder
{
protected:
	SceneBuilder() = default;
	SceneBuilder(const SceneBuilder&) = default;
	~SceneBuilder() = default;
	SceneBuilder& operator =(const SceneBuilder&) = default;

public:
	virtual void addVertices(std::vector<vertex>&& vertices) = 0;
	virtual void addSurface(PrimitiveType primitive_type, std::vector<std::uint32_t>&& indices, const char* name, size_t name_length, const char* material_name) = 0;
	virtual void addMaterial(const char* name, const math::float3& ambient, const math::float3& diffuse, const math::float4& specular, float alpha, const char* tex_name) = 0;
	virtual void addTexture(const char* name, const char* filename) = 0;
};

#endif  // INCLUDED_SCENE_BUILDER
