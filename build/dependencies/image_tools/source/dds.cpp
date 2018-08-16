


#include <cstdint>
#include <cstring>
#include <utility>
#include <stdexcept>
#include <iostream>
#include <fstream>

#include <win32/unicode.h>

#include "io.h"

#include "dds.h"


namespace
{
	struct DDS_PIXELFORMAT
	{
		std::uint32_t dwSize;
		std::uint32_t dwFlags;
		std::uint32_t dwFourCC;
		std::uint32_t dwRGBBitCount;
		std::uint32_t dwRBitMask;
		std::uint32_t dwGBitMask;
		std::uint32_t dwBBitMask;
		std::uint32_t dwABitMask;
	};

	struct DDS_HEADER
	{
		std::uint32_t dwSize;
		std::uint32_t dwFlags;
		std::uint32_t dwHeight;
		std::uint32_t dwWidth;
		std::uint32_t dwLinearSize;
		std::uint32_t dwDepth;
		std::uint32_t dwMipMapCount;
		std::uint32_t dwReserved1[11];
		DDS_PIXELFORMAT ddpf;
		std::uint32_t dwCaps;
		std::uint32_t dwCaps2;
		std::uint32_t dwCaps3;
		std::uint32_t dwCaps4;
		std::uint32_t dwReserved2;
	};


	constexpr unsigned long DDSD_CAPS = 0x00000001UL;
	constexpr unsigned long DDSD_HEIGHT = 0x00000002UL;
	constexpr unsigned long DDSD_WIDTH = 0x00000004UL;
	constexpr unsigned long DDSD_PITCH = 0x00000008UL;
	constexpr unsigned long DDSD_PIXELFORMAT = 0x00001000UL;
	constexpr unsigned long DDSD_MIPMAPCOUNT = 0x00020000UL;
	constexpr unsigned long DDSD_LINEARSIZE = 0x00080000UL;
	constexpr unsigned long DDSD_DEPTH = 0x00800000UL;

	constexpr unsigned long DDPF_RGB = 0x00000040UL;
	constexpr unsigned long DDPF_FOURCC = 0x00000004UL;
	constexpr unsigned long DDPF_ALPHAPIXELS = 0x00000001UL;
	constexpr unsigned long DDPF_RGBA = DDPF_RGB | DDPF_ALPHAPIXELS;

	constexpr unsigned long DDSCAPS_TEXTURE = 0x00001000UL;
	constexpr unsigned long DDSCAPS_COMPLEX = 0x00000008UL;
	constexpr unsigned long DDSCAPS_MIPMAP = 0x00400000UL;

	constexpr std::uint32_t make_FOURCC(std::uint8_t ch0, std::uint8_t ch1, std::uint8_t ch2, std::uint8_t ch3)
	{
		return (ch0 | (ch1 << 8) | (ch2 << 16) | (ch3 << 24));
	}


	std::istream& readSurface(std::istream& file, char* buffer, int width, int pixel_size)
	{
		return read(file, buffer, width * pixel_size);
	}

	std::istream& readSurface(std::istream& file, char* buffer, int width, int height, int pixel_size)
	{
		for (int y = height - 1; y >= 0; --y)
			read(file, buffer + y * width * pixel_size, width * pixel_size);
		return file;
	}

	std::ostream& writeSurface(std::ostream& file, const char* buffer, int width, int height, int pixel_size)
	{
		for (int y = height - 1; y >= 0; --y)
			write(file, buffer + y * width * pixel_size, width * pixel_size);
		return file;
	}

	bool isDDS(std::istream& file)
	{
		char magic_num[4];
		read<char, 4>(file, magic_num);

		if (std::strncmp(magic_num, "DDS ", 4) == 0)
			return true;

		file.seekg(0);
		return false;
	}

	image2DMipmap<RGBA8> readImage(std::istream& file)
	{
		if (!isDDS(file))
			throw std::runtime_error("not a dds file");

		DDS_HEADER header;
		read(file, &header);

		int levels = header.dwMipMapCount == 0 ? 1 : header.dwMipMapCount;
		int width = header.dwWidth;
		int height = header.dwHeight;

		image2DMipmap<RGBA8> image(width, height, levels);

		//auto dest = data(image);
		//for (int level = 0; level < levels; ++level)
		//{
		//	readSurface(file, dest, width, height);
		//	width /= 2;
		//	height /= 2;
		//}

		return std::move(image);
	}
}

namespace DDS
{
	image2DMipmap<RGBA8> loadRGBA8(const char* filename)
	{
		std::ifstream file(filename, std::ios::binary);

		if (!file)
			throw std::runtime_error("failed to open file");

		return readImage(file);
	}
}
