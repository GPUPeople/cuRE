


#ifndef INCLUDED_COM_SHARED_PTR
#define INCLUDED_COM_SHARED_PTR

#pragma once

#include <utility>


namespace COM
{
	template <class T>
	class shared_ptr
	{
		T* ptr;

		static void release(T* ptr) noexcept
		{
			if (ptr)
				ptr->Release();
		}

		static void acquire(T* ptr) noexcept
		{
			if (ptr)
				ptr->AddRef();
		}

	public:
		shared_ptr(std::nullptr_t ptr = nullptr) noexcept
			: ptr(nullptr)
		{
		}

		explicit shared_ptr(T* ptr) noexcept
			: ptr(ptr)
		{
		}

		shared_ptr(const shared_ptr& p) noexcept
			: ptr(p.ptr)
		{
			acquire(ptr);
		}

		shared_ptr(shared_ptr&& p) noexcept
			: ptr(p.ptr)
		{
			p.ptr = nullptr;
		}

		~shared_ptr()
		{
			release(ptr);
		}

		shared_ptr& operator =(const shared_ptr& p) noexcept
		{
			acquire(p.ptr);
			release(ptr);
			ptr = p.ptr;
			return *this;
		}

		shared_ptr& operator =(shared_ptr&& p) noexcept
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

		friend void swap(shared_ptr& a, shared_ptr& b) noexcept
		{
			using std::swap;
			swap(a.ptr, b.ptr);
		}
	};

	template <class T>
	inline shared_ptr<T> make_shared_ptr(T* ptr)
	{
		return shared_ptr<T>(ptr);
	}
}

#endif  // INCLUDED_COM_SHARED_PTR
