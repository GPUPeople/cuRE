


#ifndef INCLUDED_NAVIGATOR
#define INCLUDED_NAVIGATOR

#pragma once

#include <interface.h>

#include <math/matrix.h>


class INTERFACE Navigator
{
protected:
	Navigator() = default;
	Navigator(const Navigator&) = default;
	Navigator& operator =(const Navigator&) = default;
	~Navigator() = default;
public:
	enum class Button : unsigned int
	{
		LEFT = 1U,
		RIGHT = 2U,
		MIDDLE = 4U
	};

	virtual void reset() = 0;
	virtual void writeWorldToLocalTransform(math::affine_float4x4* M) const = 0;
	virtual void writeLocalToWorldTransform(math::affine_float4x4* M) const = 0;
	virtual void writePosition(math::float3* p) const = 0;
	virtual void buttonDown(Button button, int x, int y) = 0;
	virtual void buttonUp(Button button, int x, int y) = 0;
	virtual void mouseMove(int x, int y) = 0;
	virtual void mouseWheel(int delta) = 0;

};


#endif  // INCLUDED_NAVIGATOR
