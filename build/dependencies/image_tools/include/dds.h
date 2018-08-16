


#ifndef INCLUDED_DDS_FILE_FORMAT
#define INCLUDED_DDS_FILE_FORMAT

#pragma once

#include "rgba8.h"
#include "image.h"


namespace DDS
{
	image2DMipmap<RGBA8> loadRGBA8(const char* filename);
}

#endif  // INCLUDED_DDS_FILE_FORMAT
