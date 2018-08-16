#include "FirstPersonNavigator.h"

const float FirstPersonNavigator::ROTATION_ATTENUATION = 0.012f;
const float FirstPersonNavigator::WALK_ATTENUATION = 0.012f;
const float FirstPersonNavigator::PAN_ATTENUATION = 0.012f;

FirstPersonNavigator::FirstPersonNavigator(float speed, float phi, float theta, const math::float3& position)
	: speed(speed),
	  phi(phi), initialPhi(phi),
	  theta(theta), initialTheta(theta),
	  position(position), initialPosition(position),
	  drag(0U)
{
	update();
}

void FirstPersonNavigator::rotateH(float dphi)
{
	phi = math::fmod(phi + dphi, 2.0f * math::constants<float>::pi());
}

void FirstPersonNavigator::rotateV(float dtheta)
{
	theta = math::fmod(theta + dtheta, 2.0f * math::constants<float>::pi());
}

void FirstPersonNavigator::pan(float x, float y)
{
	position += x * u + y * v;
}

void FirstPersonNavigator::walk(float d)
{
	position += d * w;
}

void FirstPersonNavigator::reset()
{
	phi = initialPhi;
	theta = initialTheta;
	position = initialPosition;
}

void FirstPersonNavigator::update()
{
	float cp = math::cos(phi);
	float sp = math::sin(phi);
	float ct = math::cos(theta);
	float st = math::sin(theta);

	w = math::float3(ct * cp, st, ct * sp);
	v = math::float3(-st * cp, ct, -st * sp);
	u = cross(v, w);
}


void FirstPersonNavigator::writeWorldToLocalTransform(math::affine_float4x4* M) const
{
	*M = math::affine_float4x4(u.x, u.y, u.z, -dot(u, position),
	                           v.x, v.y, v.z, -dot(v, position),
	                           w.x, w.y, w.z, -dot(w, position));
}

void FirstPersonNavigator::writeLocalToWorldTransform(math::affine_float4x4* M) const
{
	*M = math::affine_float4x4(u.x, v.x, w.x, position.x,
	                           u.y, v.y, w.y, position.y,
	                           u.z, v.z, w.z, position.z);
}

void FirstPersonNavigator::writePosition(math::float3* p) const
{
	*p = FirstPersonNavigator::position;
}


void FirstPersonNavigator::buttonDown(Button button, int x, int y)
{
	drag |= static_cast<unsigned int>(button);
	last_pos = math::int2(x, y);
}

void FirstPersonNavigator::buttonUp(Button button, int x, int y)
{
	drag &= ~static_cast<unsigned int>(button);
}

void FirstPersonNavigator::mouseMove(int x, int y)
{
	if (drag)
	{
		math::int2 pos = math::int2(x, y);
		math::int2 d = pos - last_pos;

		if (drag & static_cast<unsigned int>(Button::LEFT))
		{
			rotateH(-d.x * ROTATION_ATTENUATION * speed);
			rotateV(-d.y * ROTATION_ATTENUATION * speed);
		}

		if (drag & static_cast<unsigned int>(Button::RIGHT))
		{
			math::int2 absd = abs(d);

			float dr = ((absd.y > absd.x) ? (d.y < 0 ? 1.0f : -1.0f) : (d.x > 0.0f ? 1.0f : -1.0f)) * math::sqrt(static_cast<float>(d.x * d.x + d.y * d.y)) * WALK_ATTENUATION * speed;
			walk(dr);
		}

		if (drag & static_cast<unsigned int>(Button::MIDDLE))
		{
			pan(-d.x * PAN_ATTENUATION * speed, d.y * PAN_ATTENUATION * speed);
		}

		update();

		last_pos = pos;
	}
}

void FirstPersonNavigator::mouseWheel(int delta)
{
	walk(delta * 0.007f);
	update();
}

void FirstPersonNavigator::save(Config& config)
{

}
