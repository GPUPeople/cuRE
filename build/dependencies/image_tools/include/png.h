


#ifndef INCLUDED_PNG_FILE_FORMAT
#define INCLUDED_PNG_FILE_FORMAT

#pragma once

#include <cstddef>
#include <tuple>

#include "rgba8.h"
#include "image.h"


namespace PNG
{
	std::tuple<std::size_t, std::size_t> readSize(const char* filename);

	image2D<RGBA8> loadRGBA8(const char* filename);
	void saveRGBA8(const char* filename, const image2D<RGBA8>& image);
}

#endif  // INCLUDED_PNG_FILE_FORMAT
