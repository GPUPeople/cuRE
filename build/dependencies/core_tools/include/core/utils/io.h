


#ifndef INCLUDED_CORE_UTILS_IO
#define INCLUDED_CORE_UTILS_IO

#pragma once

#include <iostream>


template <typename T>
T read(std::istream& in)
{
	T value;
	in.read(reinterpret_cast<char*>(&value), sizeof(T));
	return value;
}

template <typename T>
std::istream& read(T& value, std::istream& in)
{
	return in.read(reinterpret_cast<char*>(&value), sizeof(T));
}

template <typename T>
std::istream& read(T* values, std::istream& in, size_t count)
{
	return in.read(reinterpret_cast<char*>(values), sizeof(T) * count);
}

template <typename T, size_t N>
std::istream& read(T(&values)[N], std::istream& in)
{
	return in.read(reinterpret_cast<char*>(values), sizeof(values));
}

template <typename T>
std::ostream& write(std::ostream& out, T value)
{
	return out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
std::ostream& write(std::ostream& out, const T* values, size_t count)
{
	return out.write(reinterpret_cast<const char*>(values), sizeof(T) * count);
}

template <typename T, size_t N>
std::ostream& write(std::ostream& out, const T(&values)[N])
{
	return out.write(reinterpret_cast<const char*>(values), sizeof(values));
}

#endif  // INCLUDED_CORE_UTILS_IO
