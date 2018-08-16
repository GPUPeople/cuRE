


#ifndef INCLUDED_CURE_BLEND_SHADERS
#define INCLUDED_CURE_BLEND_SHADERS

#pragma once

#include <math/vector.h>
#include <math/matrix.h>


struct NoBlending
{
	__device__
	math::float4 operator ()(const math::float4& src) const
	{
		return src;
	}
};

struct AlphaBlending
{
	__device__
	math::float4 operator ()(const math::float4& src, const math::float4& dest) const
	{
		return src.w * src + (1.0f - src.w) * dest;
	}
};


struct TextureBlending
{
	__device__
	math::float4 operator ()(const math::float4& src, const math::float4& dest) const
	{
		return src + (1.0f - src.w) * dest;
	}
};

struct ClipspaceBlending
{
	__device__
	math::float4 operator ()(const math::float4& src, const math::float4& dest) const
	{
		//return 0.25f * src + 0.75f * dest;
		return src + dest;
	}
};

struct EyeCandyBlending
{
	__device__
	math::float4 operator ()(const math::float4& src, const math::float4& dest) const
	{
		return 0.25f * src + 0.75f * dest;
	}
};


template <typename T>
class SeparableBlendOp : T
{
public:
	__device__
	math::float4 operator ()(const math::float4& src, const math::float4& dest) const
	{
		return { T::operator ()(dest.x, src.x), T::operator ()(dest.y, src.y), T::operator ()(dest.z, src.z), src.w };
	}
};


struct Normal
{
	__device__ float operator ()(float c_b, float c_s) const
	{
		return c_s;
	}
};

struct FiftyFifty
{
	__device__ float operator ()(float c_b, float c_s) const
	{
		return 0.5f * c_b + 0.5f * c_s;
	}
};

struct Multiply
{
	__device__ float operator ()(float c_b, float c_s) const
	{
		return c_b * c_s;
	}
};

struct Screen
{
	__device__ float operator ()(float c_b, float c_s) const
	{
		return c_b + c_s - (c_b * c_s);
	}
};

struct Darken
{
	__device__ float operator ()(float c_b, float c_s) const
	{
		return min(c_b, c_s);
	}
};

struct Lighten
{
	__device__ float operator ()(float c_b, float c_s) const
	{
		return max(c_b, c_s);
	}
};

struct ColorDodge
{
	__device__ float operator ()(float c_b, float c_s) const
	{
		if (c_s < 1.0f)
			return min(1.0f, c_b / (1.0f - c_s));
		return 1.0f;
	}
};

struct ColorBurn
{
	__device__ float operator ()(float c_b, float c_s) const
	{
		if (c_s > 0.0f)
			return 1.0f - min(1.0f, (1.0f - c_b) / c_s);
		return 0.0f;
	}
};

struct HardLight
{
	__device__ float operator ()(float c_b, float c_s) const
	{
		if (c_s <= 0.5f)
			return Multiply()(c_b, 2.0f * c_s);
		return Screen()(c_b, 2.0f * c_s - 1.0f);
	}
};

class SoftLight
{
	__device__
	static float D(float x)
	{
		if (x <= 0.25)
			return ((16 * x - 12) * x + 4) * x;
		return sqrt(x);
	}

public:
	__device__ float operator ()(float c_b, float c_s) const
	{
		if (c_s <= 0.5f)
			return c_b - (1.0f - 2.0f * c_s) * c_b * (1 - c_b);
		return c_b + (2.0f * c_s - 1.0f) * (D(c_b) - c_b);
	}
};

struct Overlay
{
	__device__ float operator ()(float c_b, float c_s) const
	{
		return HardLight()(c_s, c_b);
	}
};

struct Difference
{
	__device__ float operator ()(float c_b, float c_s) const
	{
		return abs(c_b - c_s);
	}
};

struct Exclusion
{
	__device__ float operator ()(float c_b, float c_s) const
	{
		return c_b + c_s - 2.0f * c_b * c_s;
	}
};


//class NonSeparableBlendOp
//{
//protected:
//	static __device__ float lum(const math::float3& c)
//	{
//		return 0.3f * c.x + 0.59f * c.y + 0.11f * c.z;
//	}
//
//	static __device__ float sat(const math::float3& c)
//	{
//		return max(c.x, max(c.y, c.z)) - min(c.x, min(c.y, c.z));
//	}
//
//	static __device__ math::float3 clipColor(const math::float3& c)
//	{
//		float l = lum(c);
//		float n = min(c.x, min(c.y, c.z));
//		float x = max(c.x, min(c.y, c.z));
//
//		if (n < 0.0f)
//			return {
//				l + (((c.x - l) * l) / (l - n)),
//				l + (((c.y - l) * l) / (l - n)),
//				l + (((c.z - l) * l) / (l - n))
//			};
//		if (x > 1.0f)
//			return {
//				l + (((c.x - l) * (1.0f - l)) / (x - l)),
//				l + (((c.y - l) * (1.0f - l)) / (x - l)),
//				l + (((c.z - l) * (1.0f - l)) / (x - l))
//			};
//		return c;
//	}
//
//	static __device__ math::float3 setLum(const math::float3& c, float l)
//	{
//		float d = l - lum(c);
//		return clipColor({c.x + d, c.y + d, c.z + d});
//	}
//
//	static __device__ math::float3 setSat(const math::float3& c, float l)
//	{
//		(c.x < c.y) && (c.x < c.z)
//
//		int min = 0U;
//
//		if (c.x < c.y)
//			
//
//		if (c_max < c_mid)
//			swap(c_mid, c_max);
//
//		if (c_mid < c_min)
//			swap(c_min, c_mid);
//
//		if (c_max > c_min)
//
//	}
//};
//
//class BlendingHue : NonSeparableBlendOp
//{
//public:
//	__device__
//	math::float4 operator ()(const math::float4& src, const math::float4& dest) const
//	{
//		return{ T::operator ()(dest.x, src.x), T::operator ()(dest.y, src.y), T::operator ()(dest.z, src.z), src.w };
//	}
//};



struct WaterBlending
{
    __device__
        math::float4 operator ()(const math::float4& src, const math::float4& dest) const
    {
        return 0.5f * src + 0.5f * dest;
    }
};

__device__
static math::float3 opposite_color(const math::float3& src, const math::float3& grey_src, const math::float3& comp, const math::float3& grey_comp)
{
    math::float3 vec = normalize(grey_src - src);

    math::float3 comp_vec = comp - grey_comp;

    float vec_len = sqrt(dot(comp_vec, comp_vec));

    return grey_comp + vec_len * vec;
}

__device__
static math::float3 greypoint(const math::float3& val)
{
    const math::float3x3 toXYZ =
    {
        0.4124f, 0.3576f, 0.1805f,
        0.2126f, 0.7152f, 0.0722f,
        0.0193f, 0.1192f, 0.9505f
    };

    math::float3 xyz = (toXYZ * val);

    float Xn = 0.95047f;
    float Zn = 1.08883f;

    xyz.x = Xn*xyz.y;
    xyz.z = Zn*xyz.y;

    const math::float3x3 toRGB =
    {
        3.2406f, -1.5372f, -0.4986f,
        -0.9689f, 1.8758f, 0.0415f,
        0.0557f, -0.2040f, 1.0570f
    };

    return toRGB * xyz;
}

__device__
static float hue(const math::float3& val)
{
    return atan2(sqrt(3.f) * (val.y - val.z), 2.f * val.x - val.y - val.z);
}

__device__
static bool eq_hue(const math::float3& v1, const math::float3& grey_1, const math::float3& v2, const math::float3& grey_2, const float epsilon)
{
    //math::float3 vec1 = normalize(v1 - grey_1);
    //math::float3 vec2 = normalize(v2 - grey_2);

    //return dot(vec1, vec2) > (1.f - epsilon);

    math::float2 vec1 = { sqrtf(3) * (v1.y - v1.z), 2 * v1.x - v1.y - v1.z };
    vec1 = normalize(vec1);
    math::float2 vec2 = { sqrtf(3) * (v2.y - v2.z), 2 * v2.x - v2.y - v2.z };
    vec2 = normalize(vec2);

    return dot(vec1, vec2) > (1.f - epsilon);
}

struct BlendBlending
{
    __device__
    math::float4 operator ()(const math::float4& src, const math::float4& dest) const
    {
        //classic
        //return 0.5f * src + 0.5f * dest;

        float alpha = sin(0.2f*uniform[0]);
        alpha = alpha*alpha;

        math::float3 c1 = alpha * src.xyz();
        math::float3 c2 = (1.f - alpha) * dest.xyz();

        math::float3 cnew;

        math::float3 grey_c1 = greypoint(c1);
        math::float3 grey_c2 = greypoint(c2);

        float epsilon = 0.01f;

        if (eq_hue(c1, grey_c1, c2, grey_c2, epsilon))
        {
            cnew = c1 + c2;
        }
        else
        {
            math::float3 c2_ = opposite_color(c1, grey_c1, c2, grey_c2);
            cnew = c1 + c2_;
            math::float3 grey_cnew = greypoint(cnew);

            if (!eq_hue(c1, grey_c1, cnew, grey_cnew, epsilon))
            {
                math::float3 c1_ = opposite_color(c2, grey_c2, c1, grey_c1);
                cnew = c1_ + c2;
            }
        } 

        return math::float4(cnew, 1.f);
    }
};



//#define delta (6.f / 29.f)
//#define delta2 delta*delta
//#define delta3 delta2*delta
//
//#define Xn 0.95047f
//#define Zn 1.08883f
//
//static __device__ float f(float t)
//{
//    if (t > delta3)
//    {
//        return powf(t, 1.f / 3.f);
//    }
//    else
//    {
//        return (t / (3 * delta2)) + (4.f / 29.f);
//    }
//}
//
//static __device__  float f1(float t)
//{
//    if (t > delta)
//    {
//        return t*t*t;
//    }
//    else
//    {
//        return (3 * delta2) * (t - (4.f / 29.f));
//    }
//}
//
//static __device__ math::float3 to_lab2(const math::float3& val)
//{
//    const math::float3x3 toXYZ =
//    {
//        0.4124f, 0.3576f, 0.1805f,
//        0.2126f, 0.7152f, 0.0722f,
//        0.0193f, 0.1192f, 0.9505f
//    };
//
//    math::float3 xyz = (toXYZ * val);
//
//    float L = 1.16f * f(xyz.y) - 0.16f;
//    float a = 5.f * (f(xyz.x / Xn) - f(xyz.y));
//    float b = 2.f * (f(xyz.y) - f(xyz.z / Zn));
//
//    return { L, a, b };
//}
//
//static __device__ math::float3 to_rgb2(const math::float3& v)
//{
//    float L = v.x;
//    float a = v.y;
//    float b = v.z;
//
//    math::float3 xyz;
//
//    xyz.x = Xn * f1(((L + 0.16f) / 1.16f) + (a / 5.f));
//    xyz.y = f1((L + 0.16f) / 1.16f);
//    xyz.z = Zn * f1(((L + 0.16f) / 1.16f) - (b / 2.f));
//
//    const math::float3x3 toRGB =
//    {
//        3.2406f, -1.5372f, -0.4986f,
//        -0.9689f, 1.8758f, 0.0415f,
//        0.0557f, -0.2040f, 1.0570f
//    };
//
//    return toRGB * xyz;
//}
//
//static __device__ math::float3 opposite_color2(const math::float3& src, const math::float3& dst)
//{
//    math::float3 res;
//
//    math::float2 v = { -src.y, -src.z };
//    v = normalize(v);
//
//    float len = sqrt(dst.y*dst.y + dst.z*dst.z);
//
//    res.x = dst.x;
//    res.y = len * v.x;
//    res.z = len * v.y;
//    return res;
//}
//
//static __device__ bool eq_hue2(const math::float3& a, const math::float3& b, float epsilon)
//{
//    math::float2 v = normalize(math::float2(a.y, a.z ));
//    math::float2 w = normalize(math::float2(b.y, b.z ));
//
//    return dot(v, w) > (1.f - epsilon);
//}
//
struct IsoBlendBlending
{
//    __device__
//        math::float4 operator ()(const math::float4& src, const math::float4& dest)
//    {
//
//        math::float4 dst = dest;
//
//        //return (src.w * src + (1.f - src.w) * dst);
//
//        float alpha = src.w;
//
//        math::float3 c1_rgb = alpha * src.xyz();
//        math::float3 c2_rgb = (1.f - alpha) * dst.xyz();
//
//        math::float3 c1_lab = to_lab2(c1_rgb);
//        math::float3 c2_lab = to_lab2(c2_rgb);
//
//        math::float3 cnew_rgb;
//
//        float epsilon = 0.01f;
//
//        if (eq_hue2(c1_lab, c2_lab, epsilon))
//        {
//            cnew_rgb = c1_rgb + c2_rgb;
//        }
//        else
//        {
//            math::float3 c2_lab_ = opposite_color2(c1_lab, c2_lab);
//            math::float3 cnew_lab = { 0.5f * (c1_lab.x + c2_lab_.x), c1_lab.y + c2_lab_.y, c1_lab.z + c2_lab_.z };
//            //cnew_rgb = c1_rgb + to_rgb2(c2_lab_);
//            cnew_rgb = to_rgb2(cnew_lab);
//
//            if (!eq_hue2(c1_lab, cnew_lab, epsilon))
//            {
//                math::float3 c1_lab_ = opposite_color2(c2_lab, c1_lab);
//                math::float3 cnew_lab = { 0.5f * (c1_lab_.x + c2_lab.x), c1_lab_.y + c2_lab.y, c1_lab_.z + c2_lab.z };
//                //cnew_rgb = to_rgb2(c1_lab_) + c2_rgb;
//                cnew_rgb = to_rgb2(cnew_lab);
//            }
//        }
//
//        return math::float4(cnew_rgb, 1.f);
//    }

    __device__
    math::float4 operator ()(const math::float4& src, const math::float4& dest) const
    {

        //math::float4 dst = (dest.x == 1 && dest.y == 1 && dest.z == 1 && dest.w == 1 ? math::float4(0, 0, 0, 1) : dest);

        math::float4 dst = dest;

        return (src.w * src + (1.f - src.w) * dst);

        //float alpha = src.w;

        ////math::float3 c1 = alpha * src.xyz();
        ////math::float3 c2 = (1.f - alpha) * dst.xyz();

        //math::float3 c1 = alpha * src.xyz();
        //math::float3 c2 = (1.f - alpha) * dst.xyz();

        //math::float3 cnew;

        //math::float3 grey_c1 = greypoint(c1);
        //math::float3 grey_c2 = greypoint(c2);

        //float epsilon = 0.01f;

        //if (eq_hue(c1, grey_c1, c2, grey_c2, epsilon))
        //{
        //    cnew = c1 + c2;
        //}
        //else
        //{
        //    math::float3 c2_ = opposite_color(c1, grey_c1, c2, grey_c2);
        //    cnew = c1 + c2_;
        //    math::float3 grey_cnew = greypoint(cnew);

        //    if (!eq_hue(c1, grey_c1, cnew, grey_cnew, epsilon))
        //    {
        //        math::float3 c1_ = opposite_color(c2, grey_c2, c1, grey_c1);
        //        cnew = c1_ + c2;
        //    }
        //}

        //return math::float4(cnew, 1.f);
    }
};

#endif  // INCLUDED_CURE_BLEND_SHADERS
