


#ifndef INCLUDED_CURE_VERTEX_SHADER_INPUT
#define INCLUDED_CURE_VERTEX_SHADER_INPUT

#pragma once

#include <math/vector.h>

#include <ptx_primitives.cuh>

#include "shader.cuh"


namespace internal
{
	template <unsigned int i, unsigned int j, typename... T>
	struct PackedBufferAttribute;

	template <unsigned int i, typename... Tail>
	struct PackedBufferAttribute<i, 0, float, Tail...>
	{
		static constexpr int next_i = i;
		static constexpr int next_j = 1;

		__device__
		static inline float fetch(const float4* d)
		{
			if (VERTEX_FETCH_CS)
				return ldg_cs(d + i).x;
			return d[i].x;
		}
	};

	template <unsigned int i, typename... Tail>
	struct PackedBufferAttribute<i, 1, float, Tail...>
	{
		static constexpr int next_i = i;
		static constexpr int next_j = 2;

		__device__
		static inline float fetch(const float4* d)
		{
			if (VERTEX_FETCH_CS)
				return ldg_cs(d + i).y;
			return d[i].y;
		}
	};

	template <unsigned int i, typename... Tail>
	struct PackedBufferAttribute<i, 2, float, Tail...>
	{
		static constexpr int next_i = i;
		static constexpr int next_j = 3;

		__device__
		static inline float fetch(const float4* d)
		{
			if (VERTEX_FETCH_CS)
				return ldg_cs(d + i).z;
			else
				return d[i].z;
		}
	};

	template <unsigned int i, typename... Tail>
	struct PackedBufferAttribute<i, 3, float, Tail...>
	{
		static constexpr int next_i = i + 1;
		static constexpr int next_j = 0;

		__device__
		static inline float fetch(const float4* d)
		{
			if (VERTEX_FETCH_CS)
				return ldg_cs(d + i).w;
			else
				return d[i].w;
		}
	};

	template <unsigned int i, unsigned int j, typename... Tail>
	struct PackedBufferAttribute<i, j, int, Tail...>
	{
	private:
		typedef PackedBufferAttribute<i, j, float, Tail...> E;
	public:
		static constexpr int next_i = E::next_i;
		static constexpr int next_j = E::next_j;

		__device__
		static inline int fetch(const float4* d)
		{
			return __float_as_int(E::fetch(d));
		}
	};

	template <unsigned int i, typename... Tail>
	struct PackedBufferAttribute<i, 0, math::float2, Tail...>
	{
		static constexpr int next_i = i;
		static constexpr int next_j = 2;

		__device__
		static inline math::float2 fetch(const float4* d)
		{
			if (VERTEX_FETCH_CS)
			{
				auto v = ldg_cs(d + i);
				return { v.x, v.y };
			}

			return math::float2(d[i].x, d[i].y);
		}
	};

	template <unsigned int i, typename... Tail>
	struct PackedBufferAttribute<i, 1, math::float2, Tail...>
	{
		static constexpr int next_i = i;
		static constexpr int next_j = 3;

		__device__
		static inline math::float2 fetch(const float4* d)
		{
			if (VERTEX_FETCH_CS)
			{
				auto v = ldg_cs(d + i);
				return { v.y, v.z };
			}

			return math::float2(d[i].y, d[i].z);
		}
	};

	template <unsigned int i, typename... Tail>
	struct PackedBufferAttribute<i, 2, math::float2, Tail...>
	{
		static constexpr int next_i = i + 1;
		static constexpr int next_j = 0;

		__device__
		static inline math::float2 fetch(const float4* d)
		{
			if (VERTEX_FETCH_CS)
			{
				auto v = ldg_cs(d + i);
				return { v.z, v.w };
			}

			return math::float2(d[i].z, d[i].w);
		}
	};

	template <unsigned int i, typename... Tail>
	struct PackedBufferAttribute<i, 3, math::float2, Tail...>
	{
		static constexpr int next_i = i + 1;
		static constexpr int next_j = 1;

		__device__
		static inline math::float2 fetch(const float4* d)
		{
			if (VERTEX_FETCH_CS)
			{
				auto v0 = ldg_cs(d + i);
				auto v1 = ldg_cs(d + i + 1);
				return { v0.w, v1.x };
			}

			return math::float2(d[i].w, d[i + 1].x);
		}
	};

	template <unsigned int i, typename... Tail>
	struct PackedBufferAttribute<i, 0, math::float3, Tail...>
	{
		static constexpr int next_i = i;
		static constexpr int next_j = 3;

		__device__
		static inline math::float3 fetch(const float4* d)
		{
			if (VERTEX_FETCH_CS)
			{
				auto v = ldg_cs(d + i);
				return { v.x, v.y, v.z };
			}

			return math::float3(d[i].x, d[i].y, d[i].z);
		}
	};

	template <unsigned int i, typename... Tail>
	struct PackedBufferAttribute<i, 1, math::float3, Tail...>
	{
		static constexpr int next_i = i + 1;
		static constexpr int next_j = 0;

		__device__
		static inline math::float3 fetch(const float4* d)
		{
			if (VERTEX_FETCH_CS)
			{
				auto v = ldg_cs(d + i);
				return { v.y, v.x, v.w };
			}

			return math::float3(d[i].y, d[i].z, d[i].w);
		}
	};

	template <unsigned int i, typename... Tail>
	struct PackedBufferAttribute<i, 2, math::float3, Tail...>
	{
		static constexpr int next_i = i + 1;
		static constexpr int next_j = 1;

		__device__
		static inline math::float3 fetch(const float4* d)
		{
			if (VERTEX_FETCH_CS)
			{
				auto v0 = ldg_cs(d + i);
				auto v1 = ldg_cs(d + i + 1);
				return { v0.z, v0.w, v1.x };
			}

			return math::float3(d[i].z, d[i].w, d[i + 1].x);
		}
	};

	template <unsigned int i, typename... Tail>
	struct PackedBufferAttribute<i, 3, math::float3, Tail...>
	{
		static constexpr int next_i = i + 1;
		static constexpr int next_j = 2;

		__device__
		static inline math::float3 fetch(const float4* d)
		{
			if (VERTEX_FETCH_CS)
			{
				auto v0 = ldg_cs(d + i);
				auto v1 = ldg_cs(d + i + 1);
				return { v0.w, v1.x, v1.y };
			}

			return math::float3(d[i].w, d[i + 1].x, d[i + 1].y);
		}
	};

	template <unsigned int i, typename... Tail>
	struct PackedBufferAttribute<i, 0, math::float4, Tail...>
	{
		static constexpr int next_i = i + 1;
		static constexpr int next_j = 0;

		__device__
		static inline math::float4 fetch(const float4* d)
		{
			if (VERTEX_FETCH_CS)
			{
				auto v = ldg_cs(d + i);
				return { v.x, v.y, v.z, v.w };
			}

			return math::float4(d[i].x, d[i].y, d[i].z, d[i].w);
		}
	};

	template <unsigned int i, typename... Tail>
	struct PackedBufferAttribute<i, 1, math::float4, Tail...>
	{
		static constexpr int next_i = i + 1;
		static constexpr int next_j = 1;

		__device__
		static inline math::float4 fetch(const float4* d)
		{
			if (VERTEX_FETCH_CS)
			{
				auto v0 = ldg_cs(d + i);
				auto v1 = ldg_cs(d + i + 1);
				return { v0.y, v0.z, v0.w, v1.x };
			}

			return math::float4(d[i].y, d[i].z, d[i].w, d[i + 1].x);
		}
	};

	template <unsigned int i, typename... Tail>
	struct PackedBufferAttribute<i, 2, math::float4, Tail...>
	{
		static constexpr int next_i = i + 1;
		static constexpr int next_j = 2;

		__device__
		static inline math::float4 fetch(const float4* d)
		{
			if (VERTEX_FETCH_CS)
			{
				auto v0 = ldg_cs(d + i);
				auto v1 = ldg_cs(d + i + 1);
				return { v0.z, v0.w, v1.x, v1.y };
			}

			return math::float4(d[i].z, d[i].w, d[i + 1].x, d[i + 1].y);
		}
	};

	template <unsigned int i, typename... Tail>
	struct PackedBufferAttribute<i, 3, math::float4, Tail...>
	{
		static constexpr int next_i = i + 1;
		static constexpr int next_j = 3;

		__device__
		static inline math::float4 fetch(const float4* d)
		{
			if (VERTEX_FETCH_CS)
			{
				auto v0 = ldg_cs(d + i);
				auto v1 = ldg_cs(d + i + 1);
				return { v0.w, v1.x, v1.y, v1.z };
			}

			return math::float4(d[i].w, d[i + 1].x, d[i + 1].y, d[i + 1].z);
		}
	};
}


template <typename... Elements>
class PackedBufferAttributes;

template <>
class PackedBufferAttributes<>
{
protected:
	__device__
	PackedBufferAttributes(const float4* d)
	{
	}

	template <unsigned int i, unsigned int j>
	__device__
	void fetch(const float4* d)
	{
	}

public:
	template <typename F, typename... Args>
	__device__
	auto read(F& reader, const Args&... args) const
	{
		return reader(args...);
	}
};

template <typename T, typename... Tail>
class PackedBufferAttributes<T, Tail...> : private PackedBufferAttributes<Tail...>
{
private:
	T data;

protected:
	template <unsigned int i, unsigned int j>
	__device__
	void fetch(const float4* d)
	{
		typedef internal::PackedBufferAttribute<i, j, T, Tail...> E;
		data = E::fetch(d);
		PackedBufferAttributes<Tail...>::template fetch<E::next_i, E::next_j>(d);
	}

public:
	__device__
	PackedBufferAttributes(const float4* d)
		: PackedBufferAttributes<Tail...>(d)
	{
		fetch<0, 0>(d);
	}

	template <typename F, typename... Args>
	__device__
	auto read(F& reader, const Args&... args) const
	{
		return PackedBufferAttributes<Tail...>::read(reader, args..., data);
	}
};


template <unsigned int STRIDE>
class VertexBuffer
{
	static_assert(STRIDE % sizeof(float4) == 0, "ERROR: vertex attribute data must be 16 byte aligned");

public:
	__device__
	static const float4* attributes(unsigned int index)
	{
		return vertex_buffer + STRIDE / sizeof(float4) * index;
	}
};


template <typename VertexBuffer, typename... Elements>
struct VertexBufferAttributes : PackedBufferAttributes<Elements...>
{
	using Signature = ShaderSignature<Elements...>;

	__device__
	VertexBufferAttributes(unsigned int vertex_index)
		: PackedBufferAttributes<Elements...>(VertexBuffer::attributes(vertex_index))
	{
	}
};


template <typename Head, typename... Tail>
struct InputVertexAttributes;

template <typename Head>
struct InputVertexAttributes<Head> : private Head
{
	using Signature = typename Head::Signature;

	__device__
	InputVertexAttributes(unsigned int vertex_index)
		: Head(vertex_index)
	{
	}

	template <typename F>
	__device__
	auto read(F& reader) const
	{
		return Head::read(reader);
	}
};

//template <typename Head, typename... Tail>
//struct InputVertexAttributes : private Head, private InputVertexAttributes<Tail...>
//{
//	__device__
//	InputVertexAttributes(unsigned int vertex_index)
//		: Head(vertex_index), InputVertexAttributes<Tail...>(vertex_index)
//	{
//	}
//
//	template <typename F, typename... Args>
//	__device__
//	auto read(F reader, const Args&... args) const
//	{
//		return Head::read(reader, args..., data);
//	}
//};

#endif  // INCLUDED_CURE_VERTEX_SHADER_INPUT
