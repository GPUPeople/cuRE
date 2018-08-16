


#ifndef INCLUDED_UTILS
#define INCLUDED_UTILS

#pragma once

#ifdef __CUDA_CC__
#define UTILITY_FUNCTION __host__ __device__
#else
#define UTILITY_FUNCTION
#endif


UTILITY_FUNCTION
inline unsigned int divup(unsigned int a, unsigned int b)
{
	return (a + b - 1U) / b;
}

UTILITY_FUNCTION
inline int divup(int a, int b)
{
	return (a + b - 1) / b;
}


template <typename A, typename... T>
UTILITY_FUNCTION
A min(const A arg, const T... args)
{
	return min(arg, min(args...));
}

template <typename A, typename... T>
UTILITY_FUNCTION
A max(const A arg, const T... args)
{
	return max(arg, max(args...));
}

#endif  // INCLUDED_CURE_UTILS
