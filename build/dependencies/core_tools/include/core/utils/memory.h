


#ifndef INCLUDED_CORE_UTILS_MEMORY
#define INCLUDED_CORE_UTILS_MEMORY

#pragma once

#include <type_traits>
#include <memory>


namespace core
{
	template <typename T>
	inline auto make_unique_default()
	{
		return std::unique_ptr<T> { new T };
	}

	template <typename T>
	inline std::enable_if_t<std::is_array<T>::value && (std::extent<T>::value == 0), std::unique_ptr<T>> make_unique_default(std::size_t size)
	{
		return std::unique_ptr<T> { new std::remove_extent_t<T>[size] };
	}


	template <typename T, typename... Args>
	class delayed_unique_maker : std::tuple<Args...>
	{
	public:
		delayed_unique_maker(Args... args)
			: std::tuple<Args...>(std::forward<Args>(args)...)
		{
		}

		operator std::unique_ptr<T>() & = delete;

		operator std::unique_ptr<T>() &&
		{
			return std::apply([](Args... args) { return std::make_unique<T>(std::forward<Args>(args)...); }, static_cast<std::tuple<Args...>&&>(*this));
		}
	};

	template <typename T, typename... Args>
	inline auto delayed_make_unique(Args&&... args)
	{
		return delayed_unique_maker<T, Args&&...>(std::forward<Args>(args)...);
	}
}

#endif  // INCLUDED_CORE_UTILS_MEMORY
