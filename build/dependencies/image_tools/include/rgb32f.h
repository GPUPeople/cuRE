


#ifndef INCLUDED_COLOR_RGB32F
#define INCLUDED_COLOR_RGB32F

#pragma once


class RGB32F
{
private:
	float color[3];
public:
	RGB32F() = default;

#if _MSC_VER < 1900
	RGB32F(float r, float g, float b)
	{
		color[0] = r;
		color[1] = g;
		color[2] = b;
	}
#else
	RGB32F(float r, float g, float b)
		: color {r, g, b}
	{
	}
#endif

	template <int i>
	friend float channel(const RGB32F& color);
};

template <int i>
inline float channel(const RGB32F& color)
{
	static_assert(i >= 0 && i < 3, "invalid color channel index");
	return color.color[i];
}

#endif  // INCLUDED_COLOR_RGB32F
