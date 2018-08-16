


#include <cstring>

#include "argparse.h"


namespace
{
	bool compareOption(const char* arg, std::string_view option)
	{
		return std::strncmp(arg, option.data(), option.length()) == 0;
	}
}

bool parseBoolFlag(const char* const *& argv, std::string_view option)
{
	if (*argv == option)
		return true;
	return false;
}

bool parseStringArgument(const char*& value, const char* const *& argv, std::string_view option)
{
	if (!compareOption(*argv, option))
		return false;

	const char* startptr = *argv + option.length();

	if (*startptr)
	{
		value = startptr;
		return true;
	}

	startptr = *++argv;

	if (!*startptr)
		throw usage_error("expected argument");

	value = startptr;
	return true;
}

bool parseIntArgument(int& value, const char* const *& argv, std::string_view option)
{
	if (!compareOption(*argv, option))
		return false;

	const char* startptr = *argv + option.length();

	if (!*startptr)
	{
		startptr = *++argv;
		if (!*startptr)
			throw usage_error("expected integer argument");
	}

	char* endptr = nullptr;

	int v = std::strtol(startptr, &endptr, 10);

	if (*endptr)
		throw usage_error("argument is not an integer");

	value = v;
	return true;
}
