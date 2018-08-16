


#ifndef INCLUDED_PERSPECTIVE_CAMERA
#define INCLUDED_PERSPECTIVE_CAMERA

#pragma once

#include <math/vector.h>

#include "Camera.h"
#include "Navigator.h"

class PerspectiveCamera : public virtual Camera
{
private:
	float fov;
	float nearz;
	float farz;
	const Navigator* navigator;

public:
	PerspectiveCamera(float fov = 60.0f * math::constants<float>::pi() / 180.0f, float z_near = 0.1f, float z_far = 100.0f);

	void attach(const Navigator* navigator);
	void writeUniformBuffer(UniformBuffer* params, float aspect) const;
};

#endif  // INCLUDED_PERSPECTIVE_CAMERA
