


#ifndef INCLUDED_FIRST_PERSON_NAVIGATOR
#define INCLUDED_FIRST_PERSON_NAVIGATOR

#pragma once

#include "Navigator.h"


class Config;

class FirstPersonNavigator : public Navigator
{
private:
	static const float ROTATION_ATTENUATION;
	static const float WALK_ATTENUATION;
	static const float PAN_ATTENUATION;

	math::int2 last_pos;
	unsigned int drag;
	float initialPhi;
	float initialTheta;
	math::float3 initialPosition;
	float speed;
	float phi;
	float theta;
	math::float3 u;
	math::float3 v;
	math::float3 w;
	math::float3 position;

	void rotateH(float dphi);
	void rotateV(float dtheta);
	void pan(float u, float v);
	void walk(float d);
	void update();

public:
	FirstPersonNavigator(float speed, float phi, float theta, const math::float3& position = math::float3(0.0f, 0.0f, 0.0f));
	FirstPersonNavigator(Config& config);

	virtual void reset();
	virtual void writeWorldToLocalTransform(math::affine_float4x4* M) const;
	virtual void writeLocalToWorldTransform(math::affine_float4x4* M) const;
	virtual void writePosition(math::float3* p) const;
	virtual void buttonDown(Button button, int x, int y);
	virtual void buttonUp(Button button, int x, int y);
	virtual void mouseMove(int x, int y);
	virtual void mouseWheel(int delta);

	void save(Config& config);
};

#endif  // INCLUDED_FIRST_PERSON_NAVIGATOR
