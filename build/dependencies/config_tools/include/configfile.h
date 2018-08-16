


#ifndef INCLUDED_CONFIGFILE
#define INCLUDED_CONFIGFILE

#pragma once

#include <cstddef>
#include <exception>
#include <iosfwd>

#include <config/Database.h>

#include <configfile/Parser.h>
#include <configfile/Log.h>


namespace configfile
{
	struct read_error : std::exception
	{
		const char* what() const noexcept override;
	};

	ParserCallback& read(ParserCallback& parser, const char* begin, const char* end, const char* filename, Log& log);
	ParserCallback& read(ParserCallback& parser, const char* begin, const char* end, const char* filename);
	std::istream& read(ParserCallback& parser, std::istream& file, const char* filename, Log& log);
	std::istream& read(ParserCallback& parser, std::istream& file, const char* filename);
	ParserCallback& read(ParserCallback& parser, const char* filename, Log& log);
	ParserCallback& read(ParserCallback& parser, const char* filename);

	config::Database& read(config::Database& config, const char* begin, const char* end, const char* filename, Log& log);
	config::Database& read(config::Database& config, const char* begin, const char* end, const char* filename);
	std::istream& read(config::Database& config, std::istream& file, const char* filename, Log& log);
	std::istream& read(config::Database& config, std::istream& file, const char* filename);
	config::Database& read(config::Database& config, const char* filename, Log& log);
	config::Database& read(config::Database& config, const char* filename);

	std::ostream& write(std::ostream& file, const config::Database& config);
	void write(const char* filename, const config::Database& config);
}

#endif  // INCLUDED_CONFIGFILE
