


#ifndef INCLUDED_ARGPARSE
#define INCLUDED_ARGPARSE

#include <stdexcept>
#include <string_view>


struct usage_error : std::runtime_error
{
	std::runtime_error::runtime_error;
};


bool parseBoolFlag(const char* const *& argv, std::string_view option);
bool parseStringArgument(const char*& value, const char* const *& argv, std::string_view option);
bool parseIntArgument(int& value, const char* const *& argv, std::string_view option);

#endif  // INCLUDED_ARGPARSE
