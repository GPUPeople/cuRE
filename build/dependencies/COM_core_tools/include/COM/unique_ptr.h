


#ifndef INCLUDED_COM_UNIQUE_PTR
#define INCLUDED_COM_UNIQUE_PTR

#pragma once

#include <utility>


namespace COM
{
	template <class T>
	class unique_ptr
	{
		T* ptr;

		static void release(T* ptr) noexcept
		{
			if (ptr)
				ptr->Release();
		}

	public:
		unique_ptr(const unique_ptr& p) = delete;
		unique_ptr& operator =(const unique_ptr& p) = delete;

		unique_ptr(std::nullptr_t ptr = nullptr) noexcept
			: ptr(nullptr)
		{
		}

		explicit unique_ptr(T* ptr) noexcept
			: ptr(ptr)
		{
		}

		unique_ptr(unique_ptr&& p) noexcept
			: ptr(p.ptr)
		{
			p.ptr = nullptr;
		}

		~unique_ptr()
		{
			release(ptr);
		}

		unique_ptr& operator =(unique_ptr&& p) noexcept
		{
			using std::swap;
			swap(this->ptr, p.ptr);
			return *this;
		}

		T* operator ->() const noexcept { return ptr; }

		operator T*() const noexcept { return ptr; }

		T* release() noexcept
		{
			T* temp = ptr;
			ptr = nullptr;
			return temp;
		}

		void reset(T* ptr = nullptr) noexcept
		{
			using std::swap;
			swap(this->ptr, ptr);
			release(ptr);
		}

		friend void swap(unique_ptr& a, unique_ptr& b) noexcept
		{
			using std::swap;
			swap(a.ptr, b.ptr);
		}
	};

	template <class T>
	inline unique_ptr<T> make_unique_ptr(T* ptr)
	{
		return unique_ptr<T>(ptr);
	}
}

#endif  // INCLUDED_COM_UNIQUE_PTR
