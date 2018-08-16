


#ifndef INCLUDED_CONFIGFILE_DEFAULT_LOG
#define INCLUDED_CONFIGFILE_DEFAULT_LOG

#pragma once

#include <configfile/Log.h>


namespace configfile
{
	class DefaultLog : public virtual Log
	{
		int num_errors = 0;

	public:
		DefaultLog() = default;
		DefaultLog(const DefaultLog&) = delete;
		DefaultLog& operator =(const DefaultLog&) = delete;

		void warning(const char* message, const char* file, std::size_t line, std::ptrdiff_t column) override;
		void warning(const std::string& message, const char* file, std::size_t line, std::ptrdiff_t column) override;
		void error(const char* message, const char* file, std::size_t line, std::ptrdiff_t column) override;
		void error(const std::string& message, const char* file, std::size_t line, std::ptrdiff_t column) override;

		void throwErrors() const;
	};
}

#endif  // INCLUDED_CONFIGFILE_DEFAULT_LOG
