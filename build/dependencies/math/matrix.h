


#ifndef INCLUDED_MATH_MATRIX
#define INCLUDED_MATH_MATRIX

#pragma once

#include "math.h"
#include "vector.h"


namespace math
{
	template <typename T, unsigned int M, unsigned int N>
	class matrix;

	template <typename T>
	class matrix<T, 2U, 2U>;

	template <typename T>
	class matrix<T, 2U, 3U>;

	template <typename T>
	class matrix<T, 3U, 3U>;

	template <typename T>
	class matrix<T, 3U, 4U>;

	template <typename T>
	class matrix<T, 4U, 3U>;

	template <typename T>
	class matrix<T, 4U, 4U>;

	template <typename T, unsigned int D>
	class affine_matrix;

	template <typename T>
	class affine_matrix<T, 3>;

	template <typename T>
	class affine_matrix<T, 3>;


	template <typename T>
	class matrix<T, 2U, 2U>
	{
	public:
		static const unsigned int M = 2U;
		static const unsigned int N = 2U;
		typedef T field_type;

		T _11, _12;
		T _21, _22;

		matrix() = default;

		MATH_FUNCTION explicit matrix(T a)
		    : _11(a), _12(a), _21(a), _22(a)
		{
		}

		MATH_FUNCTION matrix(T m11, T m12, T m21, T m22)
		    : _11(m11), _12(m12), _21(m21), _22(m22)
		{
		}

		static MATH_FUNCTION matrix from_rows(const vector<T, 2U>& r1, const vector<T, 2U>& r2)
		{
			return matrix(r1.x, r1.y, r2.x, r2.y);
		}
		static MATH_FUNCTION matrix from_cols(const vector<T, 2U>& c1, const vector<T, 2U>& c2)
		{
			return matrix(c1.x, c2.x, c1.y, c2.y);
		}

		MATH_FUNCTION const vector<T, 2U> row1() const
		{
			return vector<T, 3U>(_11, _12);
		}
		MATH_FUNCTION const vector<T, 2U> row2() const
		{
			return vector<T, 3U>(_21, _22);
		}

		MATH_FUNCTION const vector<T, 2U> column1() const
		{
			return vector<T, 2U>(_11, _21);
		}
		MATH_FUNCTION const vector<T, 2U> column2() const
		{
			return vector<T, 2U>(_12, _22);
		}

		MATH_FUNCTION friend matrix transpose(const matrix& m)
		{
			return matrix(m._11, m._21, m._12, m._22);
		}

		MATH_FUNCTION static matrix scale(const vector<T, 2>& angle)
		{
			matrix M(angle.x, 0.0f, 0.0f, angle.y);
			return M;
		}

		MATH_FUNCTION friend matrix operator*(const matrix& a, const matrix& b)
		{
			return matrix(a._11 * b._11 + a._12 * b._21, a._11 * b._12 + a._12 * b._22, a._21 * b._11 + a._22 * b._21, a._21 * b._12 + a._22 * b._22);
		}

		MATH_FUNCTION friend vector<T, 2u> operator*(const matrix& a, const vector<T, 2u>& b)
		{
			return vector<T, 2u>(a._11 * b.x + a._12 * b.y,
			                     a._21 * b.x + a._22 * b.y);
		}

		MATH_FUNCTION friend matrix operator+(const matrix& a, const matrix& b)
		{
			return matrix(a._11 + b._11, a._12 + b._12, a._21 + b._21, a._22 + b._22);
		}

		MATH_FUNCTION friend matrix operator*(T f, const matrix& m)
		{
			return matrix(f * m._11, f * m._12, f * m._21, f * m._22);
		}

		MATH_FUNCTION friend matrix operator*(const matrix& m, T f)
		{
			return f * m;
		}

		MATH_FUNCTION friend T trace(const matrix& M)
		{
			return M._11 + M._22;
		}
	};



	template <typename T>
	class matrix<T, 2U, 3U>
	{
	public:
		static const unsigned int M = 2U;
		static const unsigned int N = 3U;
		typedef T field_type;

		T _11, _12, _13;
		T _21, _22, _23;

		matrix() = default;

		MATH_FUNCTION explicit matrix(T a)
			: _11(a), _12(a), _13(a), _21(a), _22(a), _23(a)
		{
		}

		MATH_FUNCTION matrix(T m11, T m12, T m13, T m21, T m22, T m23)
			: _11(m11), _12(m12), _13(m13), _21(m21), _22(m22), _23(m23)
		{
		}

		MATH_FUNCTION matrix(const affine_matrix<T, 2U>& M)
			: _11(M._11), _12(M._12), _13(M._13), _21(M._21), _22(M._22), _23(M._23)
		{
		}

		static MATH_FUNCTION matrix from_rows(const vector<T, 3U>& r1, const vector<T, 3U>& r2)
		{
			return matrix(r1.x, r1.y, r1.z, r2.x, r2.y, r2.z);
		}

		static MATH_FUNCTION matrix from_cols(const vector<T, 2U>& c1, const vector<T, 2U>& c2, const vector<T, 2U>& c3)
		{
			return matrix(c1.x, c2.x, c3.x, c1.y, c2.y, c3.y);
		}

		MATH_FUNCTION const vector<T, 3U> row1() const
		{
			return vector<T, 3U>(_11, _12, _13);
		}
		MATH_FUNCTION const vector<T, 3U> row2() const
		{
			return vector<T, 3U>(_21, _22, _23);
		}

		MATH_FUNCTION const vector<T, 2U> column1() const
		{
			return vector<T, 2U>(_11, _21);
		}
		MATH_FUNCTION const vector<T, 2U> column2() const
		{
			return vector<T, 2U>(_12, _22);
		}
		MATH_FUNCTION const vector<T, 2U> column3() const
		{
			return vector<T, 2U>(_13, _23);
		}

		MATH_FUNCTION friend matrix<T, 2U, 3U> transpose(const matrix& m)
		{
			return matrix<T, 2U, 3U>(m._11, m._21, m._12, m._22, m._13, m._23);
		}

		MATH_FUNCTION friend matrix operator+(const matrix& a, const matrix& b)
		{
			return matrix(a._11 + b._11, a._12 + b._12, a._13 + b._13, a._21 + b._21, a._22 + b._22, a._23 + b._23);
		}

		MATH_FUNCTION friend matrix operator*(float f, const matrix& m)
		{
			return matrix(f * m._11, f * m._12, f * m._13, f * m._21, f * m._22, f * m._23);
		}

		MATH_FUNCTION friend matrix operator*(const matrix& m, float f)
		{
			return f * m;
		}
	};


	template <typename T>
	class matrix<T, 3U, 3U>
	{
	public:
		static const unsigned int M = 3U;
		static const unsigned int N = 3U;
		typedef T field_type;

		T _11, _12, _13;
		T _21, _22, _23;
		T _31, _32, _33;

		matrix() = default;

		MATH_FUNCTION explicit matrix(T a)
		    : _11(a), _12(a), _13(a), _21(a), _22(a), _23(a), _31(a), _32(a), _33(a)
		{
		}

		MATH_FUNCTION matrix(T m11, T m12, T m13, T m21, T m22, T m23, T m31, T m32, T m33)
		    : _11(m11), _12(m12), _13(m13), _21(m21), _22(m22), _23(m23), _31(m31), _32(m32), _33(m33)
		{
		}

		MATH_FUNCTION matrix(const affine_matrix<T, 2U>& M)
		    : _11(M._11), _12(M._12), _13(M._13), _21(M._21), _22(M._22), _23(M._23), _31(0.0f), _32(0.0f), _33(1.0f)
		{
		}

		static MATH_FUNCTION matrix from_rows(const vector<T, 3U>& r1, const vector<T, 3U>& r2, const vector<T, 3U>& r3)
		{
			return matrix(r1.x, r1.y, r1.z, r2.x, r2.y, r2.z, r3.x, r3.y, r3.z);
		}
		static MATH_FUNCTION matrix from_cols(const vector<T, 3U>& c1, const vector<T, 3U>& c2, const vector<T, 3U>& c3)
		{
			return matrix(c1.x, c2.x, c3.x, c1.y, c2.y, c3.y, c1.z, c2.z, c3.z);
		}

		MATH_FUNCTION const vector<T, 3U> row1() const
		{
			return vector<T, 3U>(_11, _12, _13);
		}
		MATH_FUNCTION const vector<T, 3U> row2() const
		{
			return vector<T, 3U>(_21, _22, _23);
		}
		MATH_FUNCTION const vector<T, 3U> row3() const
		{
			return vector<T, 3U>(_31, _32, _33);
		}

		MATH_FUNCTION const vector<T, 3U> column1() const
		{
			return vector<T, 3U>(_11, _21, _31);
		}
		MATH_FUNCTION const vector<T, 3U> column2() const
		{
			return vector<T, 3U>(_12, _22, _32);
		}
		MATH_FUNCTION const vector<T, 3U> column3() const
		{
			return vector<T, 3U>(_13, _23, _33);
		}

		MATH_FUNCTION friend matrix transpose(const matrix& m)
		{
			return matrix(m._11, m._21, m._31, m._12, m._22, m._32, m._13, m._23, m._33);
		}

		MATH_FUNCTION friend T determinant(const matrix& m)
		{
			return m._11 * m._22 * m._33 + m._12 * m._23 * m._31 + m._13 * m._21 * m._32 - m._13 * m._22 * m._31 - m._12 * m._21 * m._33 - m._11 * m._23 * m._32;
		}

		MATH_FUNCTION friend matrix operator+(const matrix& a, const matrix& b)
		{
			return matrix(a._11 + b._11, a._12 + b._12, a._13 + b._13, a._21 + b._21, a._22 + b._22, a._23 + b._23, a._31 + b._31, a._32 + b._32, a._33 + b._33);
		}

		MATH_FUNCTION friend matrix operator*(float f, const matrix& m)
		{
			return matrix(f * m._11, f * m._12, f * m._13, f * m._21, f * m._22, f * m._23, f * m._31, f * m._32, f * m._33);
		}

		MATH_FUNCTION friend matrix operator*(const matrix& m, float f)
		{
			return f * m;
		}

		MATH_FUNCTION friend matrix operator*(const matrix& a, const matrix& b)
		{
			return matrix(a._11 * b._11 + a._12 * b._21 + a._13 * b._31, a._11 * b._12 + a._12 * b._22 + a._13 * b._32, a._11 * b._13 + a._12 * b._23 + a._13 * b._33, a._21 * b._11 + a._22 * b._21 + a._23 * b._31, a._21 * b._12 + a._22 * b._22 + a._23 * b._32, a._21 * b._13 + a._22 * b._23 + a._23 * b._33, a._31 * b._11 + a._32 * b._21 + a._33 * b._31, a._31 * b._12 + a._32 * b._22 + a._33 * b._32, a._31 * b._13 + a._32 * b._23 + a._33 * b._33);
		}

		MATH_FUNCTION friend vector<T, 3U> operator*(const vector<T, 3U>& v, const matrix& m)
		{
			return vector<T, 3U>(v.x * m._11 + v.y * m._21 + v.z * m._31,
			                     v.x * m._12 + v.y * m._22 + v.z * m._32,
			                     v.x * m._13 + v.y * m._23 + v.z * m._33);
		}

		MATH_FUNCTION friend vector<T, 3U> operator*(const matrix& m, const vector<T, 3U>& v)
		{
			return vector<T, 3U>(m._11 * v.x + m._12 * v.y + m._13 * v.z,
			                     m._21 * v.x + m._22 * v.y + m._23 * v.z,
			                     m._31 * v.x + m._32 * v.y + m._33 * v.z);
		}

		MATH_FUNCTION friend T trace(const matrix& M)
		{
			return M._11 + M._22 + M._33;
		}
	};

	template <typename T>
	class matrix<T, 3U, 4U>
	{
	public:
		static const unsigned int M = 3U;
		static const unsigned int N = 4U;
		typedef T field_type;

		T _11, _12, _13, _14;
		T _21, _22, _23, _24;
		T _31, _32, _33, _34;

		matrix() = default;

		MATH_FUNCTION explicit matrix(T a)
		    : _11(a), _12(a), _13(a), _14(a), _21(a), _22(a), _23(a), _24(a), _31(a), _32(a), _33(a), _34(a)
		{
		}

		MATH_FUNCTION matrix(T m11, T m12, T m13, T m14, T m21, T m22, T m23, T m24, T m31, T m32, T m33, T m34)
		    : _11(m11), _12(m12), _13(m13), _14(m14), _21(m21), _22(m22), _23(m23), _24(m24), _31(m31), _32(m32), _33(m33), _34(m34)
		{
		}

		MATH_FUNCTION matrix(const affine_matrix<T, 3U>& M)
		    : _11(M._11), _12(M._12), _13(M._13), _14(M._14), _21(M._21), _22(M._22), _23(M._23), _24(M._24), _31(M._31), _32(M._32), _33(M._33), _34(M._34)
		{
		}

		static MATH_FUNCTION matrix from_rows(const vector<T, 4U>& r1, const vector<T, 4U>& r2, const vector<T, 4U>& r3)
		{
			return matrix(r1.x, r1.y, r1.z, r1.w, r2.x, r2.y, r2.z, r2.w, r3.x, r3.y, r3.z, r3.w);
		}
		static MATH_FUNCTION matrix from_cols(const vector<T, 3U>& c1, const vector<T, 3U>& c2, const vector<T, 3U>& c3, const vector<T, 3U>& c4)
		{
			return matrix(c1.x, c2.x, c3.x, c4.x, c1.y, c2.y, c3.y, c4.y, c1.z, c2.z, c3.z, c4.z);
		}

		MATH_FUNCTION const vector<T, 4U> row1() const
		{
			return vector<T, 4U>(_11, _12, _13, _14);
		}
		MATH_FUNCTION const vector<T, 4U> row2() const
		{
			return vector<T, 4U>(_21, _22, _23, _24);
		}
		MATH_FUNCTION const vector<T, 4U> row3() const
		{
			return vector<T, 4U>(_31, _32, _33, _34);
		}

		MATH_FUNCTION const vector<T, 3U> column1() const
		{
			return vector<T, 3U>(_11, _21, _31);
		}
		MATH_FUNCTION const vector<T, 3U> column2() const
		{
			return vector<T, 3U>(_12, _22, _32);
		}
		MATH_FUNCTION const vector<T, 3U> column3() const
		{
			return vector<T, 3U>(_13, _23, _33);
		}
		MATH_FUNCTION const vector<T, 3U> column4() const
		{
			return vector<T, 3U>(_14, _24, _34);
		}

		MATH_FUNCTION friend matrix<T, 4U, 3U> transpose(const matrix& m)
		{
			return matrix<T, 4U, 3U>(m._11, m._21, m._31, m._12, m._22, m._32, m._13, m._23, m._33, m._14, m._24, m._34);
		}

		MATH_FUNCTION friend matrix operator+(const matrix& a, const matrix& b)
		{
			return matrix(a._11 + b._11, a._12 + b._12, a._13 + b._13, a._14 + b._14, a._21 + b._21, a._22 + b._22, a._23 + b._23, a._24 + b._24, a._31 + b._31, a._32 + b._32, a._33 + b._33, a._34 + b._34);
		}

		MATH_FUNCTION friend matrix operator*(float f, const matrix& m)
		{
			return matrix(f * m._11, f * m._12, f * m._13, f * m._14, f * m._21, f * m._22, f * m._23, f * m._24, f * m._31, f * m._32, f * m._33, f * m._34);
		}

		MATH_FUNCTION friend matrix operator*(const matrix& m, float f)
		{
			return f * m;
		}

		MATH_FUNCTION friend vector<T, 4U> operator*(const vector<T, 3U>& v, const matrix& m)
		{
			return vector<T, 4U>(v.x * m._11 + v.y * m._21 + v.z * m._31,
			                     v.x * m._12 + v.y * m._22 + v.z * m._32,
			                     v.x * m._13 + v.y * m._23 + v.z * m._33,
			                     v.x * m._14 + v.y * m._24 + v.z * m._34);
		}

		MATH_FUNCTION friend vector<T, 3U> operator*(const matrix& m, const vector<T, 4>& v)
		{
			return vector<T, 3U>(m._11 * v.x + m._12 * v.y + m._13 * v.z + m._14 * v.w,
			                     m._21 * v.x + m._22 * v.y + m._23 * v.z + m._24 * v.w,
			                     m._31 * v.x + m._32 * v.y + m._33 * v.z + m._34 * v.w);
		}
	};

	template <typename T>
	class matrix<T, 4U, 3U>
	{
	public:
		static const unsigned int M = 4U;
		static const unsigned int N = 3U;
		typedef T field_type;

		T _11, _12, _13;
		T _21, _22, _23;
		T _31, _32, _33;
		T _41, _42, _43;

		matrix() = default;

		MATH_FUNCTION explicit matrix(T a)
		    : _11(a), _12(a), _13(a), _21(a), _22(a), _23(a), _31(a), _32(a), _33(a), _41(a), _42(a), _43(a)
		{
		}

		MATH_FUNCTION matrix(T m11, T m12, T m13, T m21, T m22, T m23, T m31, T m32, T m33, T m41, T m42, T m43)
		    : _11(m11), _12(m12), _13(m13), _21(m21), _22(m22), _23(m23), _31(m31), _32(m32), _33(m33), _41(m41), _42(m42), _43(m43)
		{
		}

		static MATH_FUNCTION matrix from_rows(const vector<T, 3U>& r1, const vector<T, 3U>& r2, const vector<T, 3U>& r3, const vector<T, 3U>& r4)
		{
			return matrix(r1.x, r1.y, r1.z, r2.x, r2.y, r2.z, r3.x, r3.y, r3.z, r4.x, r4.y, r4.z);
		}
		static MATH_FUNCTION matrix from_cols(const vector<T, 4U>& c1, const vector<T, 4U>& c2, const vector<T, 4U>& c3)
		{
			return matrix(c1.x, c2.x, c3.x, c1.y, c2.y, c3.y, c1.z, c2.z, c3.z, c1.w, c2.w, c3.w);
		}

		MATH_FUNCTION const vector<T, 3U> row1() const
		{
			return vector<T, 3U>(_11, _12, _13);
		}
		MATH_FUNCTION const vector<T, 3U> row2() const
		{
			return vector<T, 3U>(_21, _22, _23);
		}
		MATH_FUNCTION const vector<T, 3U> row3() const
		{
			return vector<T, 3U>(_31, _32, _33);
		}
		MATH_FUNCTION const vector<T, 3U> row4() const
		{
			return vector<T, 3U>(_41, _42, _43);
		}

		MATH_FUNCTION const vector<T, 3U> column1() const
		{
			return vector<T, 4U>(_11, _21, _31, _41);
		}
		MATH_FUNCTION const vector<T, 3U> column2() const
		{
			return vector<T, 4U>(_12, _22, _32, _42);
		}
		MATH_FUNCTION const vector<T, 3U> column3() const
		{
			return vector<T, 4U>(_13, _23, _33, _43);
		}

		MATH_FUNCTION friend matrix<T, 3U, 4U> transpose(const matrix& m)
		{
			return matrix<T, 3U, 4U>(m._11, m._21, m._31, m._41, m._12, m._22, m._32, m._42, m._13, m._23, m._33, m._43);
		}

		MATH_FUNCTION friend matrix operator+(const matrix& a, const matrix& b)
		{
			return matrix(a._11 + b._11, a._12 + b._12, a._13 + b._13, a._21 + b._21, a._22 + b._22, a._23 + b._23, a._31 + b._31, a._32 + b._32, a._33 + b._33, a._41 + b._41, a._42 + b._42, a._43 + b._43);
		}

		MATH_FUNCTION friend matrix operator*(float f, const matrix& m)
		{
			return matrix(f * m._11, f * m._12, f * m._13, f * m._21, f * m._22, f * m._23, f * m._31, f * m._32, f * m._33, f * m._41, f * m._42, f * m._43);
		}

		MATH_FUNCTION friend matrix operator*(const matrix& m, float f)
		{
			return f * m;
		}

		MATH_FUNCTION friend vector<T, 3U> operator*(const vector<T, 4U>& v, const matrix& m)
		{
			return vector<T, 3U>(v.x * m._11 + v.y * m._21 + v.z * m._31 + v.w * m._41,
			                     v.x * m._12 + v.y * m._22 + v.z * m._32 + v.w * m._42,
			                     v.x * m._13 + v.y * m._23 + v.z * m._33 + v.w * m._43);
		}

		MATH_FUNCTION friend vector<T, 4U> operator*(const matrix& m, const vector<T, 3>& v)
		{
			return vector<T, 4U>(m._11 * v.x + m._12 * v.y + m._13 * v.z,
			                     m._21 * v.x + m._22 * v.y + m._23 * v.z,
			                     m._31 * v.x + m._32 * v.y + m._33 * v.z,
			                     m._41 * v.x + m._42 * v.y + m._43 * v.z);
		}
	};

	template <typename T>
	class matrix<T, 4U, 4U>
	{
	public:
		static const unsigned int M = 4U;
		static const unsigned int N = 4U;
		typedef T field_type;

		union
		{
			struct
			{
				T _11, _12, _13, _14;
				T _21, _22, _23, _24;
				T _31, _32, _33, _34;
				T _41, _42, _43, _44;
			};
			T _m[16];
		};

		matrix() = default;

		MATH_FUNCTION explicit matrix(T a)
		    : _11(a), _12(a), _13(a), _14(a), _21(a), _22(a), _23(a), _24(a), _31(a), _32(a), _33(a), _34(a), _41(a), _42(a), _43(a), _44(a)
		{
		}

		MATH_FUNCTION matrix(T m11, T m12, T m13, T m14, T m21, T m22, T m23, T m24, T m31, T m32, T m33, T m34, T m41, T m42, T m43, T m44)
		    : _11(m11), _12(m12), _13(m13), _14(m14), _21(m21), _22(m22), _23(m23), _24(m24), _31(m31), _32(m32), _33(m33), _34(m34), _41(m41), _42(m42), _43(m43), _44(m44)
		{
		}

		MATH_FUNCTION matrix(const affine_matrix<T, 3U>& M)
		    : _11(M._11), _12(M._12), _13(M._13), _14(M._14), _21(M._21), _22(M._22), _23(M._23), _24(M._24), _31(M._31), _32(M._32), _33(M._33), _34(M._34), _41(0.0f), _42(0.0f), _43(0.0f), _44(1.0f)
		{
		}

		static MATH_FUNCTION matrix from_rows(const vector<T, 4U>& r1, const vector<T, 4U>& r2, const vector<T, 4U>& r3, const vector<T, 4U>& r4)
		{
			return matrix(r1.x, r1.y, r1.z, r1.w, r2.x, r2.y, r2.z, r2.w, r3.x, r3.y, r3.z, r3.w, r4.x, r4.y, r4.z, r4.w);
		}
		static MATH_FUNCTION matrix from_cols(const vector<T, 4U>& c1, const vector<T, 4U>& c2, const vector<T, 4U>& c3, const vector<T, 4U>& c4)
		{
			return matrix(c1.x, c2.x, c3.x, c4.x, c1.y, c2.y, c3.y, c4.y, c1.z, c2.z, c3.z, c4.z, c1.w, c2.w, c3.w, c4.w);
		}

		MATH_FUNCTION const vector<T, 4U> row1() const
		{
			return vector<T, 4U>(_11, _12, _13, _14);
		}
		MATH_FUNCTION const vector<T, 4U> row2() const
		{
			return vector<T, 4U>(_21, _22, _23, _24);
		}
		MATH_FUNCTION const vector<T, 4U> row3() const
		{
			return vector<T, 4U>(_31, _32, _33, _34);
		}
		MATH_FUNCTION const vector<T, 4U> row4() const
		{
			return vector<T, 4U>(_41, _42, _43, _44);
		}

		MATH_FUNCTION const vector<T, 4U> column1() const
		{
			return vector<T, 4U>(_11, _21, _31, _41);
		}
		MATH_FUNCTION const vector<T, 4U> column2() const
		{
			return vector<T, 4U>(_12, _22, _32, _42);
		}
		MATH_FUNCTION const vector<T, 4U> column3() const
		{
			return vector<T, 4U>(_13, _23, _33, _43);
		}
		MATH_FUNCTION const vector<T, 4U> column4() const
		{
			return vector<T, 4U>(_14, _24, _34, _44);
		}

		MATH_FUNCTION friend matrix transpose(const matrix& m)
		{
			return matrix(m._11, m._21, m._31, m._41, m._12, m._22, m._32, m._42, m._13, m._23, m._33, m._43, m._14, m._24, m._34, m._44);
		}

		MATH_FUNCTION friend matrix operator+(const matrix& a, const matrix& b)
		{
			return matrix(a._11 + b._11, a._12 + b._12, a._13 + b._13, a._14 + b._14, a._21 + b._21, a._22 + b._22, a._23 + b._23, a._24 + b._24, a._31 + b._31, a._32 + b._32, a._33 + b._33, a._34 + b._34, a._41 + b._41, a._42 + b._42, a._43 + b._43, a._44 + b._44);
		}

		MATH_FUNCTION friend matrix operator*(float f, const matrix& m)
		{
			return matrix(f * m._11, f * m._12, f * m._13, f * m._14, f * m._21, f * m._22, f * m._23, f * m._24, f * m._31, f * m._32, f * m._33, f * m._34, f * m._41, f * m._42, f * m._43, f * m._44);
		}

		MATH_FUNCTION friend matrix operator*(const matrix& m, float f)
		{
			return f * m;
		}

		MATH_FUNCTION friend matrix operator*(const matrix& a, const matrix& b)
		{
			return matrix(a._11 * b._11 + a._12 * b._21 + a._13 * b._31 + a._14 * b._41, a._11 * b._12 + a._12 * b._22 + a._13 * b._32 + a._14 * b._42, a._11 * b._13 + a._12 * b._23 + a._13 * b._33 + a._14 * b._43, a._11 * b._14 + a._12 * b._24 + a._13 * b._34 + a._14 * b._44, a._21 * b._11 + a._22 * b._21 + a._23 * b._31 + a._24 * b._41, a._21 * b._12 + a._22 * b._22 + a._23 * b._32 + a._24 * b._42, a._21 * b._13 + a._22 * b._23 + a._23 * b._33 + a._24 * b._43, a._21 * b._14 + a._22 * b._24 + a._23 * b._34 + a._24 * b._44, a._31 * b._11 + a._32 * b._21 + a._33 * b._31 + a._34 * b._41, a._31 * b._12 + a._32 * b._22 + a._33 * b._32 + a._34 * b._42, a._31 * b._13 + a._32 * b._23 + a._33 * b._33 + a._34 * b._43, a._31 * b._14 + a._32 * b._24 + a._33 * b._34 + a._34 * b._44, a._41 * b._11 + a._42 * b._21 + a._43 * b._31 + a._44 * b._41, a._41 * b._12 + a._42 * b._22 + a._43 * b._32 + a._44 * b._42, a._41 * b._13 + a._42 * b._23 + a._43 * b._33 + a._44 * b._43, a._41 * b._14 + a._42 * b._24 + a._43 * b._34 + a._44 * b._44);
		}

		MATH_FUNCTION friend vector<T, 4U> operator*(const vector<T, 4U>& v, const matrix& m)
		{
			return vector<T, 4U>(v.x * m._11 + v.y * m._21 + v.z * m._31 + v.w * m._41,
			                     v.x * m._12 + v.y * m._22 + v.z * m._32 + v.w * m._42,
			                     v.x * m._13 + v.y * m._23 + v.z * m._33 + v.w * m._43,
			                     v.x * m._14 + v.y * m._24 + v.z * m._34 + v.w * m._44);
		}

		MATH_FUNCTION friend vector<T, 4U> operator*(const matrix& m, const vector<T, 4>& v)
		{
			return vector<T, 4U>(m._11 * v.x + m._12 * v.y + m._13 * v.z + m._14 * v.w,
			                     m._21 * v.x + m._22 * v.y + m._23 * v.z + m._24 * v.w,
			                     m._31 * v.x + m._32 * v.y + m._33 * v.z + m._34 * v.w,
			                     m._41 * v.x + m._42 * v.y + m._43 * v.z + m._44 * v.w);
		}

		MATH_FUNCTION T operator[](unsigned int i) const
		{
			return _m[i];
		}

		MATH_FUNCTION friend T trace(const matrix& M)
		{
			return M._11 + M._22 + M._33 + M._44;
		}
	};

	template <typename T>
	class affine_matrix<T, 2U>
	{
	public:
		typedef T field_type;

		T _11, _12, _13;
		T _21, _22, _23;

		affine_matrix() = default;

		MATH_FUNCTION explicit affine_matrix(T a)
		    : _11(a), _12(a), _13(a), _21(a), _22(a), _23(a)
		{
		}

		MATH_FUNCTION affine_matrix(T m11, T m12, T m13, T m21, T m22, T m23)
		    : _11(m11), _12(m12), _13(m13), _21(m21), _22(m22), _23(m23)
		{
		}

		MATH_FUNCTION affine_matrix(const matrix<T, 2U, 3U>& M)
		    : _11(M._11), _12(M._12), _13(M._13), _21(M._21), _22(M._22), _23(M._23)
		{
		}

		MATH_FUNCTION affine_matrix(const matrix<T, 3U, 3U>& M)
		    : _11(M._11), _12(M._12), _13(M._13), _21(M._21), _22(M._22), _23(M._23)
		{
		}

		MATH_FUNCTION friend affine_matrix operator+(const affine_matrix& a, const affine_matrix& b)
		{
			return matrix<T, 3U, 3U>(a) + matrix<T, 3U, 3U>(b);
		}

		MATH_FUNCTION friend affine_matrix operator+(const affine_matrix& a, const matrix<T, 3U, 3U>& b)
		{
			return matrix<T, 3U, 3U>(a) + b;
		}

		MATH_FUNCTION friend affine_matrix operator+(const matrix<T, 3U, 3U>& a, const affine_matrix& b)
		{
			return a + matrix<T, 3U, 3U>(b);
		}

		MATH_FUNCTION friend affine_matrix operator*(const affine_matrix& a, const affine_matrix& b)
		{
			return matrix<T, 3U, 3U>(a) * matrix<T, 3U, 3U>(b);
		}

		MATH_FUNCTION friend const matrix<T, 3U, 3U> operator*(const affine_matrix& a, const matrix<T, 3U, 3U>& b)
		{
			return matrix<T, 3U, 3U>(a) * b;
		}

		MATH_FUNCTION friend const matrix<T, 3U, 3U> operator*(const matrix<T, 3U, 3U>& a, const affine_matrix& b)
		{
			return a * matrix<T, 3U, 3U>(b);
		}

		MATH_FUNCTION friend vector<T, 3U> operator*(const vector<T, 3U>& v, const affine_matrix& m)
		{
			return vector<T, 3U>(v.x * m._11 + v.y * m._21,
			                     v.x * m._12 + v.y * m._22,
			                     v.x * m._13 + v.y * m._23 + v.z);
		}

		MATH_FUNCTION friend vector<T, 3U> operator*(const affine_matrix& m, const vector<T, 3>& v)
		{
			return vector<T, 4U>(m._11 * v.x + m._12 * v.y + m._13 * v.z,
			                     m._21 * v.x + m._22 * v.y + m._23 * v.z,
			                     v.z);
		}
	};

	template <typename T>
	class affine_matrix<T, 3U>
	{
	public:
		typedef T field_type;

		T _11, _12, _13, _14;
		T _21, _22, _23, _24;
		T _31, _32, _33, _34;

		affine_matrix() = default;

		MATH_FUNCTION explicit affine_matrix(T a)
		    : _11(a), _12(a), _13(a), _14(a), _21(a), _22(a), _23(a), _24(a), _31(a), _32(a), _33(a), _34(a)
		{
		}

		MATH_FUNCTION affine_matrix(T m11, T m12, T m13, T m14, T m21, T m22, T m23, T m24, T m31, T m32, T m33, T m34)
		    : _11(m11), _12(m12), _13(m13), _14(m14), _21(m21), _22(m22), _23(m23), _24(m24), _31(m31), _32(m32), _33(m33), _34(m34)
		{
		}

		MATH_FUNCTION affine_matrix(const matrix<T, 3U, 4U>& M)
		    : _11(M._11), _12(M._12), _13(M._13), _14(M._14), _21(M._21), _22(M._22), _23(M._23), _24(M._24), _31(M._31), _32(M._32), _33(M._33), _34(M._34)
		{
		}

		MATH_FUNCTION affine_matrix(const matrix<T, 4U, 4U>& M)
		    : _11(M._11), _12(M._12), _13(M._13), _14(M._14), _21(M._21), _22(M._22), _23(M._23), _24(M._24), _31(M._31), _32(M._32), _33(M._33), _34(M._34)
		{
		}

		MATH_FUNCTION friend affine_matrix operator+(const affine_matrix& a, const affine_matrix& b)
		{
			return matrix<T, 4U, 4U>(a) + matrix<T, 4U, 4U>(b);
		}

		MATH_FUNCTION friend affine_matrix operator+(const affine_matrix& a, const matrix<T, 4U, 4U>& b)
		{
			return matrix<T, 4U, 4U>(a) + b;
		}

		MATH_FUNCTION friend affine_matrix operator+(const matrix<T, 4U, 4U>& a, const affine_matrix& b)
		{
			return a + matrix<T, 4U, 4U>(b);
		}

		MATH_FUNCTION friend affine_matrix operator*(const affine_matrix& a, const affine_matrix& b)
		{
			return matrix<T, 4U, 4U>(a) * matrix<T, 4U, 4U>(b);
		}

		MATH_FUNCTION friend const matrix<T, 4U, 4U> operator*(const affine_matrix& a, const matrix<T, 4U, 4U>& b)
		{
			return matrix<T, 4U, 4U>(a) * b;
		}

		MATH_FUNCTION friend const matrix<T, 4U, 4U> operator*(const matrix<T, 4U, 4U>& a, const affine_matrix& b)
		{
			return a * matrix<T, 4U, 4U>(b);
		}

		MATH_FUNCTION friend vector<T, 4U> operator*(const vector<T, 4U>& v, const affine_matrix& m)
		{
			return vector<T, 4U>(v.x * m._11 + v.y * m._21 + v.z * m._31,
			                     v.x * m._12 + v.y * m._22 + v.z * m._32,
			                     v.x * m._13 + v.y * m._23 + v.z * m._33,
			                     v.x * m._14 + v.y * m._24 + v.z * m._34 + v.w);
		}

		MATH_FUNCTION friend vector<T, 4U> operator*(const affine_matrix& m, const vector<T, 4>& v)
		{
			return vector<T, 4U>(m._11 * v.x + m._12 * v.y + m._13 * v.z + m._14 * v.w,
			                     m._21 * v.x + m._22 * v.y + m._23 * v.z + m._24 * v.w,
			                     m._31 * v.x + m._32 * v.y + m._33 * v.z + m._34 * v.w,
			                     v.w);
		}
	};

	template <typename T>
	MATH_FUNCTION inline T det(const matrix<T, 2U, 2U>& m)
	{
		return m._11 * m._22 - m._21 * m._12;
	}

	template <typename T>
	MATH_FUNCTION inline T det(const matrix<T, 3U, 3U>& m)
	{
		return m._11 * det(matrix<T, 2U, 2U>(m._22, m._23, m._32, m._33)) -
		       m._12 * det(matrix<T, 2U, 2U>(m._21, m._23, m._31, m._33)) +
		       m._13 * det(matrix<T, 2U, 2U>(m._21, m._22, m._31, m._32));
	}

	template <typename T>
	MATH_FUNCTION inline T det(const matrix<T, 4U, 4U>& m)
	{
		return m._11 * det(matrix<T, 3U, 3U>(m._22, m._23, m._24, m._32, m._33, m._34, m._42, m._43, m._44)) -
		       m._12 * det(matrix<T, 3U, 3U>(m._21, m._23, m._24, m._31, m._33, m._34, m._41, m._43, m._44)) +
		       m._13 * det(matrix<T, 3U, 3U>(m._21, m._22, m._24, m._31, m._32, m._34, m._41, m._42, m._44)) -
		       m._14 * det(matrix<T, 3U, 3U>(m._21, m._22, m._23, m._31, m._32, m._33, m._41, m._42, m._43));
	}

	template <typename T>
	MATH_FUNCTION inline matrix<T, 3U, 3U> adj(const matrix<T, 3U, 3U>& m)
	{
		return transpose(matrix<T, 3U, 3U>(
		    det(matrix<T, 2U, 2U>(m._22, m._23, m._32, m._33)),
		    -det(matrix<T, 2U, 2U>(m._21, m._23, m._31, m._33)),
		    det(matrix<T, 2U, 2U>(m._21, m._22, m._31, m._32)),

		    -det(matrix<T, 2U, 2U>(m._12, m._13, m._32, m._33)),
		    det(matrix<T, 2U, 2U>(m._11, m._13, m._31, m._33)),
		    -det(matrix<T, 2U, 2U>(m._11, m._12, m._31, m._32)),

		    det(matrix<T, 2U, 2U>(m._12, m._13, m._22, m._23)),
		    -det(matrix<T, 2U, 2U>(m._11, m._13, m._21, m._23)),
		    det(matrix<T, 2U, 2U>(m._11, m._12, m._21, m._22))));
	}

	template <typename T>
	MATH_FUNCTION inline matrix<T, 4U, 4U> adj(const matrix<T, 4U, 4U>& m)
	{
		return transpose(matrix<T, 4U, 4U>(
		    det(matrix<T, 3U, 3U>(m._22, m._23, m._24, m._32, m._33, m._34, m._42, m._43, m._44)),
		    -det(matrix<T, 3U, 3U>(m._21, m._23, m._24, m._31, m._33, m._34, m._41, m._43, m._44)),
		    det(matrix<T, 3U, 3U>(m._21, m._22, m._24, m._31, m._32, m._34, m._41, m._42, m._44)),
		    -det(matrix<T, 3U, 3U>(m._21, m._22, m._23, m._31, m._32, m._33, m._41, m._42, m._43)),

		    -det(matrix<T, 3U, 3U>(m._12, m._13, m._14, m._32, m._33, m._34, m._42, m._43, m._44)),
		    det(matrix<T, 3U, 3U>(m._11, m._13, m._14, m._31, m._33, m._34, m._41, m._43, m._44)),
		    -det(matrix<T, 3U, 3U>(m._11, m._12, m._14, m._31, m._32, m._34, m._41, m._42, m._44)),
		    det(matrix<T, 3U, 3U>(m._11, m._12, m._13, m._31, m._32, m._33, m._41, m._42, m._43)),

		    det(matrix<T, 3U, 3U>(m._12, m._13, m._14, m._22, m._23, m._24, m._42, m._43, m._44)),
		    -det(matrix<T, 3U, 3U>(m._11, m._13, m._14, m._21, m._23, m._24, m._41, m._43, m._44)),
		    det(matrix<T, 3U, 3U>(m._11, m._12, m._14, m._21, m._22, m._24, m._41, m._42, m._44)),
		    -det(matrix<T, 3U, 3U>(m._11, m._12, m._13, m._21, m._22, m._23, m._41, m._42, m._43)),

		    -det(matrix<T, 3U, 3U>(m._12, m._13, m._14, m._22, m._23, m._24, m._32, m._33, m._34)),
		    det(matrix<T, 3U, 3U>(m._11, m._13, m._14, m._21, m._23, m._24, m._31, m._33, m._34)),
		    -det(matrix<T, 3U, 3U>(m._11, m._12, m._14, m._21, m._22, m._24, m._31, m._32, m._34)),
		    det(matrix<T, 3U, 3U>(m._11, m._12, m._13, m._21, m._22, m._23, m._31, m._32, m._33))));
	}

	template <typename T, unsigned int N>
	MATH_FUNCTION inline matrix<T, N, N> inverse(const matrix<T, N, N>& M)
	{
		// TODO: optimize; compute det using adj
		return rcp(det(M)) * adj(M);
	}

	template <typename T, unsigned int D>
	MATH_FUNCTION inline affine_matrix<T, D> inverse(const affine_matrix<T, D>& M)
	{
		return affine_matrix<T, D>(inverse(matrix<T, D + 1, D + 1>(M)));
	}

	typedef matrix<float, 2U, 2U> float2x2;
	typedef matrix<float, 2U, 3U> float2x3;
	typedef matrix<float, 3U, 3U> float3x3;
	typedef matrix<float, 3U, 4U> float3x4;
	typedef matrix<float, 4U, 3U> float4x3;
	typedef matrix<float, 4U, 4U> float4x4;

	typedef affine_matrix<float, 2U> affine_float3x3;
	typedef affine_matrix<float, 3U> affine_float4x4;

	template <typename T>
	MATH_FUNCTION inline T identity();

	template <>
	MATH_FUNCTION inline float2x2 identity<float2x2>()
	{
		return float2x2(1.0f, 0.0f, 0.0f, 1.0f);
	}

	template <>
	MATH_FUNCTION inline float3x3 identity<float3x3>()
	{
		return float3x3(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
	}

	template <>
	MATH_FUNCTION inline math::affine_float3x3 identity<math::affine_float3x3>()
	{
		return math::affine_float3x3(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	}

	template <>
	MATH_FUNCTION inline math::float4x4 identity<math::float4x4>()
	{
		return math::float4x4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
	}

	template <>
	MATH_FUNCTION inline math::affine_float4x4 identity<math::affine_float4x4>()
	{
		return math::affine_float4x4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	}
}

//using math::float2x3;
//using math::float3x3;
//using math::math::float4x4;

#endif // INCLUDED_MATH_MATRIX
