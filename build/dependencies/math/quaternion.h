


#ifndef INCLUDED_MATH_QUATERNION
#define INCLUDED_MATH_QUATERNION

#pragma once

#include "vector.h"
#include "matrix.h"


namespace math
{
	template <typename T>
	class quaternion
	{
	public:
		typedef T field_type;

		T x;
		T y;
		T z;
		T w;

		MATH_FUNCTION quaternion() : x(0), y(0), z(0), w(1)
		{
		}

		MATH_FUNCTION explicit quaternion(T a)
		    : x(a), y(a), z(a), w(a)
		{
		}

		MATH_FUNCTION quaternion(T x, T y, T z, T w)
		    : x(x), y(y), z(z), w(w)
		{
		}

		MATH_FUNCTION quaternion operator-() const
		{
			return quaternion(-x, -y, -z, -w);
		}

		MATH_FUNCTION quaternion& operator+=(const quaternion& v)
		{
			x += v.x;
			y += v.y;
			z += v.z;
			w += v.w;
			return *this;
		}

		MATH_FUNCTION quaternion& operator-=(const quaternion& v)
		{
			x -= v.x;
			y -= v.y;
			z -= v.z;
			w -= v.w;
			return *this;
		}

		MATH_FUNCTION quaternion& operator*=(T a)
		{
			x *= a;
			y *= a;
			z *= a;
			w *= a;
			return *this;
		}

		MATH_FUNCTION quaternion& operator*=(const quaternion& q)
		{
			quaternion t(x * q.w + y * q.z - z * q.y + w * q.x,
			             -x * q.z + y * q.w + z * q.x + w * q.y,
			             x * q.y - y * q.x + z * q.w + w * q.z,
			             -x * q.x - y * q.y - z * q.z + w * q.w);
			*this = t;
			return *this;
		}

		MATH_FUNCTION friend inline const quaternion operator+(const quaternion& a, const quaternion& b)
		{
			return quaternion(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
		}

		MATH_FUNCTION friend inline const quaternion operator-(const quaternion& a, const quaternion& b)
		{
			return quaternion(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
		}

		MATH_FUNCTION friend inline const quaternion operator*(T a, const quaternion& v)
		{
			return quaternion(a * v.x, a * v.y, a * v.z, a * v.w);
		}

		MATH_FUNCTION friend inline const quaternion operator*(const quaternion& v, T a)
		{
			return a * v;
		}

		MATH_FUNCTION friend inline const quaternion operator*(const quaternion& a, const quaternion& b)
		{
			return quaternion(a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x,
			                  -a.x * b.z + a.y * b.w + a.z * b.x + a.w * b.y,
			                  a.x * b.y - a.y * b.x + a.z * b.w + a.w * b.z,
			                  -a.x * b.x - a.y * b.y - a.z * b.z + a.w * b.w);
		}

		MATH_FUNCTION friend inline quaternion conjugate(const quaternion& q)
		{
			return quaternion(-q.x, -q.y, -q.z, q.w);
		}

		MATH_FUNCTION friend inline T length(const quaternion& q)
		{
			return sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
		}

		MATH_FUNCTION friend inline quaternion normalize(const quaternion& q)
		{
			return q * rcp(length(q));
		}

		MATH_FUNCTION static inline quaternion rotateX(T angle)
		{
			T s = std::sin(angle / T(2));
			T c = std::cos(angle / T(2));
			return quaternion(s, 0, 0, c);
		}
		MATH_FUNCTION static inline quaternion rotateY(T angle)
		{
			T s = std::sin(angle / T(2));
			T c = std::cos(angle / T(2));
			return quaternion(0, s, 0, c);
		}
		MATH_FUNCTION static inline quaternion rotateZ(T angle)
		{
			T s = std::sin(angle / T(2));
			T c = std::cos(angle / T(2));
			return quaternion(0, 0, s, c);
		}

		MATH_FUNCTION static inline quaternion rotateXYZ(const vector<T, 3U>& angles)
		{
			T sa = std::sin(angles.x / T(2));
			T sb = std::sin(angles.y / T(2));
			T sg = std::sin(angles.z / T(2));
			T ca = std::cos(angles.x / T(2));
			T cb = std::cos(angles.y / T(2));
			T cg = std::cos(angles.z / T(2));
			return quaternion(sa * sb * cg + ca * cb * sg,
			                  sa * cb * cg + ca * sb * sg,
			                  ca * sb * cg - sa * cb * sg,
			                  ca * cb * cg - sa * sb * sg);
		}

		MATH_FUNCTION static inline quaternion rotateAxis(vector<T, 3U> axis, T angle)
		{
			T s = std::sin(angle / T(2));
			T c = std::cos(angle / T(2));
			axis = normalize(axis);
			return quaternion(axis.x * s, axis.y * s, axis.z * s, c);
		}

		MATH_FUNCTION inline const vector<T, 3U> rotate(const vector<T, 3U>& p) const
		{
			quaternion i(y * p.z - z * p.y + w * p.x,
			             -x * p.z + z * p.x + w * p.y,
			             x * p.y - y * p.x + w * p.z,
			             -x * p.x - y * p.y - z * p.z);

			return vector<T, 3U>(i.x * w - i.y * z + i.z * y - i.w * x,
			                     i.x * z + i.y * w - i.z * x - i.w * y,
			                     -i.x * y + i.y * x + i.z * w - i.w * z);
		}
		MATH_FUNCTION inline const vector<T, 4U> rotate(const vector<T, 4U>& p) const
		{
			quaternion i(y * p.z - z * p.y + w * p.x,
			             -x * p.z + z * p.x + w * p.y,
			             x * p.y - y * p.x + w * p.z,
			             -x * p.x - y * p.y - z * p.z);

			return vector<T, 4U>(i.x * w - i.y * z + i.z * y - i.w * x,
			                     i.x * z + i.y * w - i.z * x - i.w * y,
			                     -i.x * y + i.y * x + i.z * w - i.w * z,
			                     1);
		}
		MATH_FUNCTION inline const vector<T, 3U> inv_rotate(const vector<T, 3U>& p) const
		{
			quaternion i(-y * p.z + z * p.y + w * p.x,
			             x * p.z - z * p.x + w * p.y,
			             -x * p.y + y * p.x + w * p.z,
			             x * p.x + y * p.y + z * p.z);

			return vector<T, 3U>(i.x * w + i.y * z - i.z * y + i.w * x,
			                     -i.x * z + i.y * w + i.z * x + i.w * y,
			                     i.x * y - i.y * x + i.z * w + i.w * z);
		}
		MATH_FUNCTION inline const vector<T, 4U> inv_rotate(const vector<T, 4U>& p) const
		{
			quaternion i(-y * p.z + z * p.y + w * p.x,
			             x * p.z - z * p.x + w * p.y,
			             -x * p.y + y * p.x + w * p.z,
			             x * p.x + y * p.y + z * p.z);

			return vector<T, 4U>(i.x * w + i.y * z - i.z * y + i.w * x,
			                     -i.x * z + i.y * w + i.z * x + i.w * y,
			                     i.x * y - i.y * x + i.z * w + i.w * z,
			                     1);
		}

		friend inline std::ostream& operator>>(std::ostream& stream, const quaternion& q)
		{
			stream << q.x << ", ";
			stream << q.y << ", ";
			stream << q.z << ", ";
			stream << q.w << ", ";
			return stream;
		}

		friend inline std::istream& operator>>(std::istream& stream, quaternion& q)
		{
			char c;
			stream >> q.x;
			stream >> c;
			if (c != ',')
				return stream;
			stream >> q.y;
			stream >> c;
			if (c != ',')
				return stream;
			stream >> q.z;
			stream >> c;
			if (c != ',')
				return stream;
			stream >> q.w;
			stream >> c;
			if (c != ',')
				return stream;
			return stream;
		}
	};

	template <typename T>
	MATH_FUNCTION T identity();

	template <>
	MATH_FUNCTION inline quaternion<float> identity<quaternion<float> >()
	{
		return quaternion<float>(0.0f, 0.0f, 0.0f, 1.0f);
	}

	template <>
	MATH_FUNCTION inline quaternion<double> identity<quaternion<double> >()
	{
		return quaternion<double>(0.0, 0.0, 0.0, 1.0);
	}

	template <>
	MATH_FUNCTION inline quaternion<long double> identity<quaternion<long double> >()
	{
		return quaternion<long double>(0.0l, 0.0l, 0.0l, 1.0l);
	}

	template <typename T>
	MATH_FUNCTION matrix<T, 3, 3> rotationMatrix3(const quaternion<T>& q)
	{
		return matrix<T, 3, 3>(T(1) - T(2) * q.y * q.y - T(2) * q.z * q.z, T(2) * q.x * q.y - T(2) * q.z * q.w, T(2) * q.x * q.z + T(2) * q.y * q.w, T(2) * q.x * q.y + T(2) * q.z * q.w, T(1) - T(2) * q.x * q.x - T(2) * q.z * q.z, T(2) * q.y * q.z - T(2) * q.x * q.w, T(2) * q.x * q.z - T(2) * q.y * q.w, T(2) * q.y * q.z + T(2) * q.x * q.w, T(1) - T(2) * q.x * q.x - T(2) * q.y * q.y);
	}

	template <typename T>
	MATH_FUNCTION matrix<T, 4, 4> rotationMatrix4(const quaternion<T>& q)
	{
		return matrix<T, 4, 4>(T(1) - T(2) * q.y * q.y - T(2) * q.z * q.z, T(2) * q.x * q.y - T(2) * q.z * q.w, T(2) * q.x * q.z + T(2) * q.y * q.w, T(0), T(2) * q.x * q.y + T(2) * q.z * q.w, T(1) - T(2) * q.x * q.x - T(2) * q.z * q.z, T(2) * q.y * q.z - T(2) * q.x * q.w, T(0), T(2) * q.x * q.z - T(2) * q.y * q.w, T(2) * q.y * q.z + T(2) * q.x * q.w, T(1) - T(2) * q.x * q.x - T(2) * q.y * q.y, T(0), T(0), T(0), T(0), T(1));
	}

	template <typename T>
	MATH_FUNCTION quaternion<T> quaternionFromRotationMatrix(const matrix<T, 3U, 3U>& M)
	{
		T t = M._11 + M._22 + M._33 + math::constants<T>::one();

		if (t <= math::constants<T>::zero())
		{
			if (M._11 > M._22 && M._11 > M._33)
			{
				T S = sqrt(math::constants<T>::one() + M._11 - M._22 - M._33);
				S += S;

				T a = rcp(S);

				return normalize(quaternion<T>(half(half(S)), (M._12 - M._21) * a, (M._13 - M._31) * a, (M._32 - M._23) * a));
			}
			else if (M._22 > M._33)
			{
				T S = sqrt(math::constants<T>::one() - M._11 + M._22 - M._33);
				S += S;

				T a = rcp(S);

				return normalize(quaternion<T>((M._13 - M._31) * a, half(half(S)), (M._23 - M._32) * a, (M._13 - M._31) * a));
			}
			else
			{
				T S = sqrt(math::constants<T>::one() - M._11 - M._22 + M._33);
				S += S;

				T a = rcp(S);

				return normalize(quaternion<T>((M._13 - M._31) * a, (M._23 - M._32) * a, half(half(S)), (M._21 - M._12) * a));
			}
		}

		T S = sqrt(t);
		S += S;

		auto a = rcp(S);

		return normalize(quaternion<T>((M._32 - M._23) * a, (M._13 - M._31) * a, (M._21 - M._12) * a, half(half(S))));
	}

	template <typename T>
	MATH_FUNCTION quaternion<T> nlerp(const quaternion<T>& a, const quaternion<T>& b, T t)
	{
		return normalize(quaternion<T>(lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t), lerp(a.w, b.w, t)));
	}

	typedef quaternion<float> quat4f;
}

#endif // INCLUDED_MATH_MATRIX
