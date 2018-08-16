


#ifndef INCLUDED_NONCAMERA
#define INCLUDED_NONCAMERA

#pragma once

#include <math/vector.h>

#include "Camera.h"
#include "Navigator.h"

class Noncamera : public virtual Camera
{
public:
	void attach(const Navigator* navigator);
	void writeUniformBuffer(UniformBuffer* params, float aspect) const;
};

#endif  // INCLUDED_NONCAMERA
