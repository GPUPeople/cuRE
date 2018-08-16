


#ifndef INCLUDED_RESOURCE
#define INCLUDED_RESOURCE

#pragma once

#include <interface.h>

#include <math/vector.h>


class RGBA8;


struct INTERFACE Resource
{
	virtual void destroy() = 0;

protected:
	Resource() = default;
	Resource(const Resource&) = default;
	Resource& operator =(const Resource&) = default;
	~Resource() = default;
};

struct INTERFACE Geometry : public virtual Resource
{
	virtual void draw() const = 0;
	virtual void draw(int start, int num_indices) const = 0;

protected:
	Geometry() = default;
	Geometry(const Geometry&) = default;
	Geometry& operator =(const Geometry&) = default;
	~Geometry() = default;
};

struct INTERFACE Material : public virtual Resource
{
	virtual void draw(const Geometry* geometry) const = 0;
	virtual void draw(const Geometry* geometry, int start, int num_indices) const = 0;

protected:
	Material() = default;
	Material(const Material&) = default;
	Material& operator =(const Material&) = default;
	~Material() = default;
};

struct INTERFACE Texture : public virtual Resource
{
	virtual Material* createTexturedMaterial(const math::float4& color) = 0;
	virtual Material* createTexturedLitMaterial(const math::float4& color) = 0;

protected:
	Texture() = default;
	Texture(const Texture&) = default;
	Texture& operator =(const Texture&) = default;
	~Texture() = default;
};

#endif  // INCLUDED_RESOURCE
