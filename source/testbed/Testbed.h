

#ifndef INCLUDED_TESTBED
#define INCLUDED_TESTBED

#pragma once

#include <iosfwd>

#include "Config.h"

class Testbed
{
public:
	Testbed();

	std::ostream& listDevices(std::ostream& out) const;
	int run(Config& config, int res_x, int res_y, int record, const char* perf_file);
};

#endif // INCLUDED_TESTBED
