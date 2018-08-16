


#ifndef INCLUDED_CONFIGFILE_PARSER
#define INCLUDED_CONFIGFILE_PARSER

#pragma once

#include <stdexcept>

#include <configfile/Stream.h>


namespace configfile
{
	class parse_error : public std::runtime_error
	{
	public:
		parse_error(const std::string& msg)
			: runtime_error(msg)
		{
		}
	};


	struct INTERFACE ParserCallback
	{
		virtual void addString(const std::string& key, const std::string& value) = 0;
		virtual void addInt(const std::string& key, int value) = 0;
		virtual void addFloat(const std::string& key, float value) = 0;
		virtual void addTuple(const std::string& key, std::vector<std::string> value) = 0;
		virtual ParserCallback* enterNode(const std::string& key) = 0;
		virtual void leaveNode() = 0;

	protected:
		ParserCallback() = default;
		ParserCallback(ParserCallback&&) = default;
		ParserCallback(const ParserCallback&) = default;
		ParserCallback& operator =(ParserCallback&&) = default;
		ParserCallback& operator =(const ParserCallback&) = default;
		~ParserCallback() = default;
	};

	Stream& parse(ParserCallback& callback, Stream& stream);
}


#endif  // INCLUDED_CONFIGFILE_PARSER
