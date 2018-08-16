


#ifndef INCLUDED_ORBITAL_NAVIGATOR
#define INCLUDED_ORBITAL_NAVIGATOR

#pragma once

#include "Navigator.h"


class Config;

class OrbitalNavigator : public Navigator
{
private:
	math::float3 u;
	math::float3 v;
	math::float3 w;
	math::float3 position;

	math::int2 last_pos;
	unsigned int drag;

	float phi;
	float theta;
	float radius;
	math::float3 lookat;

	math::float3 initial_lookat;
	float initial_phi;
	float initial_theta;
	float initial_radius;

	void rotateH(float dphi);
	void rotateV(float dtheta);
	void zoom(float dr);
	void pan(float u, float v);
	void update();

public:
	OrbitalNavigator(Config& config, float phi, float theta, float radius, const math::float3& lookat = math::float3(0.0f, 0.0f, 0.0f));

	virtual void reset();
	virtual void writeWorldToLocalTransform(math::affine_float4x4* M) const;
	virtual void writeLocalToWorldTransform(math::affine_float4x4* M) const;
	virtual void writePosition(math::float3* p) const;
	virtual void buttonDown(Button button, int x, int y);
	virtual void buttonUp(Button button, int x, int y);
	virtual void mouseMove(int x, int y);
	virtual void mouseWheel(int delta);

	void save(Config& config) const;
};

#endif  // INCLUDED_ORBITAL_NAVIGATOR
