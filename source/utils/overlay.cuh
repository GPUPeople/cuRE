


#ifndef INCLUDED_OVERLAY
#define INCLUDED_OVERLAY

#pragma once

//#include <memory>


template <typename T>
struct inplace_destruct_delete
{
	__device__
	void operator ()(T* ptr) const
	{
		ptr->~T();
	}
};



template <typename... Types>
union overlay;

template <>
union overlay<>
{
	template <typename F>
	__device__
	void project(F) = delete;
};

template <typename T, typename... Tail>
union overlay<T, Tail...>
{
	T o;
	overlay<Tail...> tail;

	template <typename R>
	__device__
	auto project(R(&f)(const T&)) const -> decltype(f(o))
	{
		return f(o);
	}

	template <typename R>
	__device__
	auto project(R(&f)(T&)) -> decltype(f(o))
	{
		return f(o);
	}

	template <typename R, typename P>
	__device__
	auto project(R(&f)(const P&)) const -> decltype(tail.project(f))
	{
		return tail.project(f);
	}

	template <typename R, typename P>
	__device__
	auto project(R(&f)(P&)) -> decltype(tail.project(f))
	{
		return tail.project(f);
	}
};


//template <typename T, typename... Types, typename... Args>
//std::unique_ptr<T, inplace_destruct_delete<T>> construct(overlay<Types...>& overlay, Args&&... args)
//{
//	return new(&static_cast<T&>(overlay)) T(args...);
//}

#endif  // INCLUDED_OVERLAY
