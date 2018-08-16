


#ifndef INCLUDED_COLOR_RGBA32F
#define INCLUDED_COLOR_RGBA32F

#pragma once


class RGBA32F
{
private:
	float color[4];
public:
	RGBA32F() = default;

#if _MSC_VER < 1900
	RGBA32F(float r, float g, float b, float a = 1.0f)
	{
		color[0] = r;
		color[1] = g;
		color[2] = b;
		color[3] = a;
	}
#else
	RGBA32F(float r, float g, float b, float a = 1.0f)
		: color {r, g, b, a}
	{
	}
#endif

	template <int i>
	friend float channel(const RGBA32F& color);
};

template <int i>
inline float channel(const RGBA32F& color)
{
	static_assert(i >= 0 && i < 4, "invalid color channel index");
	return color.color[i];
}

#endif  // INCLUDED_COLOR_RGBA32F
