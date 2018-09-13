


#ifndef INCLUDED_CONFIGFILE_LOG
#define INCLUDED_CONFIGFILE_LOG

#pragma once

#include <cstddef>
#include <string>

#include <core/interface>


namespace configfile
{
	struct INTERFACE Log
	{
		virtual void warning(const char* message, const char* file, std::size_t line, std::ptrdiff_t column) = 0;
		virtual void warning(const std::string& message, const char* file, std::size_t line, std::ptrdiff_t column) = 0;
		virtual void error(const char* message, const char* file, std::size_t line, std::ptrdiff_t column) = 0;
		virtual void error(const std::string& message, const char* file, std::size_t line, std::ptrdiff_t column) = 0;

	protected:
		Log() = default;
		Log(const Log&) = default;
		~Log() = default;
		Log& operator =(const Log&) = default;
	};
}

#endif  // INCLUDED_CONFIGFILE_LOG
