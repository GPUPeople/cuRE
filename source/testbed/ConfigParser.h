


#ifndef INCLUDED_CONFIGFILE_PARSER
#define INCLUDED_CONFIGFILE_PARSER

#pragma once

#include <stdexcept>

#include "ConfigStream.h"


namespace ConfigFile
{
	class parse_error : public std::runtime_error
	{
	public:
		parse_error(const std::string& msg)
			: runtime_error(msg)
		{
		}
	};


	class INTERFACE ParserCallback
	{
	protected:
		ParserCallback() = default;
		ParserCallback(const ParserCallback&) = default;
		~ParserCallback() = default;
		ParserCallback& operator =(const ParserCallback&) = default;
	public:
		virtual void addString(const std::string& key, const std::string& value) = 0;
		virtual void addInt(const std::string& key, int value) = 0;
		virtual void addFloat(const std::string& key, float value) = 0;
		virtual void addTuple(const std::string& key, std::vector<std::string> value) = 0;
		virtual ParserCallback& addConfig(const std::string& key) = 0;
	};

	Stream& parse(Stream& stream, ParserCallback& callback);
}


#endif  // INCLUDED_CONFIGFILE_PARSER
