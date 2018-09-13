


#include <stdexcept>
#include <string>
#include <fstream>

#include <rgb32f.h>

#include <core/utils/io>

#include "pfm.h"


namespace
{
	template <typename T>
	image2D<T> load(std::istream& file, const char* type)
	{
		std::string magic;
		size_t w;
		size_t h;
		float a;
		file >> magic >> w >> h >> a;

		if (magic != type || a != -1.0f || file.get() != '\n')
			throw std::runtime_error("unsupported file format");

		image2D<T> img(w, h);

		for (size_t j = 0; j < h; ++j)
			read(data(img) + w * (h - 1 - j), file, w);

		return img;
	}

	template <typename T>
	std::ostream& save(std::ostream& file, const image2D<T>& img, const char* type)
	{
		auto w = width(img);
		auto h = height(img);

		file << type << '\n'
		     << w << ' ' << h << '\n'
		     << -1.0f << '\n';

		for (size_t j = 0; j < h; ++j)
			write(file, data(img) + w * (h - j - 1), w);

		return file;
	}
}

namespace PFM
{
	image2D<float> loadR32F(const char* filename)
	{
		std::ifstream file(filename, std::ios::binary);

		if (!file)
			throw std::runtime_error("failed to open file");

		return ::load<float>(file, "Pf");
	}

	void saveR32F(const char* filename, const image2D<float>& img)
	{
		std::ofstream file(filename, std::ios::binary);

		if (!file)
			throw std::runtime_error("failed to open file");

		::save(file, img, "Pf");
	}

	image2D<RGB32F> loadRGB32F(const char* filename)
	{
		std::ifstream file(filename, std::ios::binary);

		if (!file)
			throw std::runtime_error("failed to open file");

		return ::load<RGB32F>(file, "PF");
	}

	void saveRGB32F(const char* filename, const image2D<RGB32F>& img)
	{
		std::ofstream file(filename, std::ios::binary);

		if (!file)
			throw std::runtime_error("failed to open file");

		::save(file, img, "PF");
	}
}
