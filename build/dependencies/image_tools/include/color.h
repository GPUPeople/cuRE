


#ifndef INCLUDED_COLOR
#define INCLUDED_COLOR

#pragma once


template <typename C>
inline auto r(const C& color) -> decltype(channel<0>(color))
{
	return channel<0>(color);
}

template <typename C>
inline auto g(const C& color) -> decltype(channel<1>(color))
{
	return channel<1>(color);
}

template <typename C>
inline auto b(const C& color) -> decltype(channel<2>(color))
{
	return channel<2>(color);
}

template <typename C>
inline auto a(const C& color) -> decltype(channel<3>(color))
{
	return channel<3>(color);
}

#endif  // INCLUDED_COLOR
