


#ifndef INCLUDED_WIN32_UNIQUE_HANDLE
#define INCLUDED_WIN32_UNIQUE_HANDLE

#pragma once

#include <utility>


namespace Win32
{
	template <typename T, T NULL_VALUE, class Deleter>
	class unique_handle : private Deleter
	{
		T h;

		void close(T handle) noexcept
		{
			if (handle != NULL_VALUE)
				Deleter::operator ()(handle);
		}

	public:
		using handle_type = T;
		using deleter_type = Deleter;

		//static constexpr T null_value = NULL_VALUE;


		unique_handle(const unique_handle&) = delete;
		unique_handle& operator =(const unique_handle&) = delete;
		
		unique_handle() noexcept
			: h(NULL_VALUE)
		{
		}

		explicit unique_handle(T handle) noexcept
			: h(handle)
		{
		}

		unique_handle(T handle, const Deleter& d) noexcept
			: Deleter(d),
			  h(handle)
		{
		}

		unique_handle(T handle, Deleter&& d) noexcept
			: Deleter(std::move(d)),
			  h(handle)
		{
		}

		unique_handle(unique_handle&& h) noexcept
			: Deleter(std::move(static_cast<Deleter&&>(h))),
			  h(h.h)
		{
			h.h = NULL_VALUE;
		}

		~unique_handle()
		{
			close(h);
		}

		operator T() const noexcept { return h; }

		unique_handle& operator =(unique_handle&& h) noexcept
		{
			using std::swap;
			swap(*this, h);
			return *this;
		}

		T release() noexcept
		{
			T temp = h;
			h = NULL_VALUE;
			return temp;
		}

		void reset(T handle = NULL_VALUE) noexcept
		{
			using std::swap;
			swap(this->h, handle);
			close(handle);
		}

		friend void swap(unique_handle& a, unique_handle& b) noexcept
		{
			using std::swap;
			swap(a.h, b.h);
		}
	};
}

#endif  // INCLUDED_WIN32_UNIQUE_HANDLE
