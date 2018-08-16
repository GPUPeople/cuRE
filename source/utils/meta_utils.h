


#ifndef INCLUDED_META_UTILS
#define INCLUDED_META_UTILS

#pragma once


template <typename... T>
struct TypeList;


template <typename A, typename B>
struct is_same
{
	static const bool value = false;
};

template <typename A>
struct is_same<A, A>
{
	static const bool value = true;
};


template <unsigned int a, unsigned int... b>
struct static_max;

template <unsigned int a>
struct static_max<a>
{
	static const unsigned int value = a;
};

template <unsigned int a, unsigned int... b>
struct static_max
{
	static const unsigned int value = a > static_max<b...>::value ? a : static_max<b...>::value;
};


template <unsigned int a, unsigned int... b>
struct static_min;

template <unsigned int a>
struct static_min<a>
{
	static const unsigned int value = a;
};

template <unsigned int a, unsigned int... b>
struct static_min
{
	static const unsigned int value = static_min<a, static_min<b...>::value>::value;
};


template <typename T, typename Type, typename... Types>
struct element_of;

template <typename T, typename... Tail>
struct element_of<T, T, Tail...>
{
	static const bool value = true;
};

template <typename T, typename Type>
struct element_of<T, Type>
{
	static const bool value = false;
};

template <typename T, typename Type, typename... Types>
struct element_of
{
	static const bool value = element_of<T, Types...>::value;
};


template <unsigned int a, unsigned int b>
struct static_divup
{
	static const unsigned int value = (a + b - 1U) / b;
};


template <bool condition, typename T, typename F>
struct conditional
{
	typedef T type;
};

template <typename T, typename F>
struct conditional<false, T, F>
{
	typedef F type;
};


template <typename T>
struct identity { typedef T type; };

template <typename T>
struct void_t { typedef void type; };

template <typename T>
struct remove_reference { typedef T type; };
template <typename T>
struct remove_reference<T&> { typedef T type; };
template <typename T>
struct remove_reference<T&&> { typedef T type; };

template <typename T>
struct remove_pointer { typedef T type; };
template <typename T>
struct remove_pointer<T*> { typedef T type; };
template <typename T>
struct remove_pointer<T* const> { typedef T type; };
template <typename T>
struct remove_pointer<T* volatile> { typedef T type; };
template <typename T>
struct remove_pointer<T* const volatile> { typedef T type; };


template <typename T>
struct add_lvalue_reference { typedef T& type; };

template <typename T>
struct add_lvalue_reference<T&&> { typedef T& type; };

template <typename T>
struct add_rvalue_reference { typedef T&& type; };

template <typename T>
struct add_rvalue_reference<T&> { typedef T& type; };

template <typename T>
typename add_rvalue_reference<T>::type declval();


template <bool B, typename T = void>
struct enable_if;

template <typename T>
struct enable_if<true, T>
{
	typedef T type;
};


template <int i, typename ... Types>
struct choose_type;

template <typename T, typename ... Types>
struct choose_type<0, T, Types...>
{
	typedef T type;
};

template <int i, typename T, typename ... Types>
struct choose_type<i, T, Types...>
{
	typedef typename choose_type<i-1, Types...>::type type;
};

template<int X>
struct static_popcnt
{
	static const int value = ((X & 0x1) + static_popcnt< (X >> 1) >::value);
};
template<>
struct static_popcnt<0>
{
	static const int value = 0;
};

template<unsigned int X, int Completed = 0>
struct static_clz
{
	static const int value = (X & 0x80000000) ? Completed : static_clz< (X << 1), Completed + 1 >::value;
};
template<unsigned int X>
struct static_clz<X, 32>
{
	static const int value = 32;
};



template<unsigned int X, unsigned int Y = 2, bool completed = false>
struct static_small_sqrt_floor
{
	static const int value = Y * Y > X ? Y - 1 : static_small_sqrt_floor<X, Y + 1, ((Y+1) * (Y+1) > X)>::value;
};
template<unsigned int X, unsigned int Y>
struct static_small_sqrt_floor<X, Y, true>
{
	static const int value = Y - 1;
};


template <typename T>
typename add_rvalue_reference<T>::type declval();

#endif  // INCLUDED_META_UTILS
