


#ifndef INCLUDED_GL_UNIQUE_NAME
#define INCLUDED_GL_UNIQUE_NAME

#pragma once

#include <utility>

#include <GL/gl.h>


namespace GL
{
	template <typename Namespace>
	class unique_name
	{
		GLuint name;
		
	public:
		unique_name(const unique_name&) = delete;
		unique_name& operator =(const unique_name&) = delete;
		
		unique_name() noexcept
			: name(Namespace::gen())
		{
		}
		
		explicit unique_name(GLuint name) noexcept
			: name(name)
		{
		}
		
		unique_name(unique_name&& n) noexcept
			: name(n.name)
		{
			n.name = 0U;
		}
		
		~unique_name()
		{
			if (name != 0U)
				Namespace::del(name);
		}
		
		operator GLuint() const noexcept { return name; }
		
		unique_name& operator =(unique_name&& n) noexcept
		{
			using std::swap;
			swap(this->name, n.name);
			return *this;
		}
		
		void reset(GLuint name = 0U) noexcept
		{
			using std::swap;
			swap(this->name, name);
			
			if (name != 0U)
				Namespace::del(name);
		}
		
		GLuint release() noexcept
		{
			GLuint name = this->name;
			this->name = 0U;
			return name;
		}
		
		friend void swap(unique_name& a, unique_name& b) noexcept
		{
			using std::swap;
			swap(a.name, b.name);
		}
	};
}

#endif  // INCLUDED_GL_UNIQUE_NAME
