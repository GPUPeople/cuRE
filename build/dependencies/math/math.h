


#ifndef INCLUDED_MATH
#define INCLUDED_MATH

#pragma once

#ifndef __CUDACC__
#include <cmath>
#include <algorithm>
#endif

#ifdef __CUDACC__
#define MATH_FUNCTION __host__ __device__
#else
#define MATH_FUNCTION
#endif


namespace math
{
	template <typename T>
	struct constants;

	template <>
	struct constants<float>
	{
		MATH_FUNCTION static float one() { return 1.0f; }
		MATH_FUNCTION static float zero() { return 0.0f; }
		MATH_FUNCTION static float pi() { return 3.1415926535897932384626434f; }
		MATH_FUNCTION static float e() { return 2.7182818284590452353602875f; }
		MATH_FUNCTION static float sqrtHalf() { return 0.70710678118654752440084436210485f; }
		MATH_FUNCTION static float sqrtTwo() { return 1.4142135623730950488016887242097f; }
		MATH_FUNCTION static float epsilon() { return 0.00000001f; }
	};

	template <>
	struct constants<double>
	{
		MATH_FUNCTION static double one() { return 1.0; }
		MATH_FUNCTION static double zero() { return 0.0; }
		MATH_FUNCTION static double pi() { return 3.1415926535897932384626434; }
		MATH_FUNCTION static double e() { return 2.7182818284590452353602875; }
		MATH_FUNCTION static double sqrtHalf() { return 0.70710678118654752440084436210485; }
		MATH_FUNCTION static double sqrtTwo() { return 1.4142135623730950488016887242097; }
		MATH_FUNCTION static double epsilon() { return 0.00000000001; }
	};

	template <>
	struct constants<long double>
	{
		MATH_FUNCTION static long double one() { return 1.0l; }
		MATH_FUNCTION static long double zero() { return 0.0l; }
		MATH_FUNCTION static long double pi() { return 3.1415926535897932384626434l; }
		MATH_FUNCTION static long double e() { return 2.7182818284590452353602875l; }
		MATH_FUNCTION static long double sqrtHalf() { return 0.70710678118654752440084436210485l; }
		MATH_FUNCTION static long double sqrtTwo() { return 1.4142135623730950488016887242097l; }
		MATH_FUNCTION static long double epsilon() { return 0.0000000000001l; }
	};

#ifndef __CUDACC__

	using std::min;
	using std::max;

	using std::abs;

	using std::exp;
	using std::frexp;
	using std::ldexp;
	using std::log;
	using std::log10;
	using std::modf;

	using std::cos;
	using std::sin;
	using std::tan;
	using std::acos;
	using std::asin;
	using std::atan;
	using std::atan2;
	using std::cosh;
	using std::sinh;
	using std::tanh;

	using std::pow;
	using std::sqrt;

	using std::floor;
	using std::ceil;

	using std::fmod;

#else
	MATH_FUNCTION inline float min(float a, float b)
	{
		return fminf(a, b);
	}
	MATH_FUNCTION inline float max(float a, float b)
	{
		return fmaxf(a, b);
	}
	MATH_FUNCTION inline float abs(float a)
	{
		return fabsf(a);
	}

	MATH_FUNCTION inline float exp(float a)
	{
		return expf(a);
	}
	MATH_FUNCTION inline float frexp(float a, int* b)
	{
		return frexpf(a, b);
	}
	MATH_FUNCTION inline float ldexp(float a, int b)
	{
		return ldexpf(a, b);
	}
	MATH_FUNCTION inline float log(float a)
	{
		return logf(a);
	}
	MATH_FUNCTION inline float log10(float a)
	{
		return log10f(a);
	}
	MATH_FUNCTION inline float modf(float a, float* b)
	{
		return modff(a, b);
	}

	MATH_FUNCTION inline float cos(float a)
	{
		return cosf(a);
	}
	MATH_FUNCTION inline float sin(float a)
	{
		return sinf(a);
	}
	MATH_FUNCTION inline float tan(float a)
	{
		return tanf(a);
	}
	MATH_FUNCTION inline float acos(float a)
	{
		return acosf(a);
	}
	MATH_FUNCTION inline float asin(float a)
	{
		return asinf(a);
	}
	MATH_FUNCTION inline float atan(float a)
	{
		return atanf(a);
	}
	MATH_FUNCTION inline float atan2(float a)
	{
		return expf(a);
	}
	MATH_FUNCTION inline float cosh(float a)
	{
		return coshf(a);
	}
	MATH_FUNCTION inline float sinh(float a)
	{
		return sinhf(a);
	}
	MATH_FUNCTION inline float tanh(float a)
	{
		return expf(a);
	}

	MATH_FUNCTION inline float pow(float a, float b)
	{
		return powf(a, b);
	}
	MATH_FUNCTION inline float sqrt(float a)
	{
		return sqrtf(a);
	}

	MATH_FUNCTION inline float floor(float a)
	{
		return floorf(a);
	}
	MATH_FUNCTION inline float ceil(float a)
	{
		return ceilf(a);
	}

	MATH_FUNCTION inline float fmod(float a, float b)
	{
		return fmodf(a, b);
	}
#endif

	template <typename T>
	MATH_FUNCTION inline T clamp(T v, T min = constants<T>::zero(), T max = constants<T>::one())
	{
		return static_cast<T>(math::min(math::max(v, min), max));
	}

	MATH_FUNCTION inline float saturate(float v)
	{
		return clamp(v, 0.0f, 1.0f);
	}

	MATH_FUNCTION inline double saturate(double v)
	{
		return clamp(v, 0.0, 1.0);
	}

	MATH_FUNCTION inline long double saturate(long double v)
	{
		return clamp(v, 0.0l, 1.0l);
	}

	MATH_FUNCTION inline float rcp(float v)
	{
		return 1.0f / v;
	}

	MATH_FUNCTION inline double rcp(double v)
	{
		return 1.0 / v;
	}

	MATH_FUNCTION inline long double rcp(long double v)
	{
		return 1.0l / v;
	}

	MATH_FUNCTION inline float frac(float v)
	{
		return v - floor(v);
	}

	MATH_FUNCTION inline double frac(double v)
	{
		return v - floor(v);
	}

	MATH_FUNCTION inline long double frac(long double v)
	{
		return v - floor(v);
	}

	MATH_FUNCTION inline float half(float v)
	{
		return v * 0.5f;
	}

	MATH_FUNCTION inline double half(double v)
	{
		return v * 0.5;
	}

	MATH_FUNCTION inline long double half(long double v)
	{
		return v * 0.5l;
	}

	MATH_FUNCTION inline float lerp(float a, float b, float t)
	{
		return (1.0f - t) * a + t * b;
	}

	MATH_FUNCTION inline double lerp(double a, double b, double t)
	{
		return (1.0 - t) * a + t * b;
	}

	MATH_FUNCTION inline long double lerp(long double a, long double b, long double t)
	{
		return (1.0l - t) * a + t * b;
	}

	MATH_FUNCTION inline float smoothstep(float t)
	{
		return t * t * (3.0f - 2.0f * t);
	}

	MATH_FUNCTION inline double smoothstep(double t)
	{
		return t * t * (3.0 - 2.0 * t);
	}

	MATH_FUNCTION inline long double smoothstep(long double t)
	{
		return t * t * (3.0l - 2.0l * t);
	}

	MATH_FUNCTION inline float smootherstep(float t)
	{
		return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
	}

	MATH_FUNCTION inline double smootherstep(double t)
	{
		return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
	}

	MATH_FUNCTION inline long double smootherstep(long double t)
	{
		return t * t * t * (t * (t * 6.0l - 15.0l) + 10.0l);
	}
}

#endif // INCLUDED_MATH
