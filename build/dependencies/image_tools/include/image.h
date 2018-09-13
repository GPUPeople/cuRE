


#ifndef INCLUDED_IMAGE
#define INCLUDED_IMAGE

#pragma once

#include <cstddef>
#include <memory>
#include <algorithm>


template <typename T>
class image2D
{
	std::unique_ptr<T[]> img;

	std::size_t width;
	std::size_t height;

	static constexpr std::size_t size(std::size_t width, std::size_t height)
	{
		return width * height;
	}

	static auto alloc(std::size_t width, std::size_t height)
	{
		return std::unique_ptr<T[]> { new T[size(width, height)] };
	}

public:
	friend std::size_t width(const image2D& img) noexcept
	{
		return img.width;
	}

	friend std::size_t height(const image2D& img) noexcept
	{
		return img.height;
	}

	friend const T* data(const image2D& img) noexcept
	{
		return &img.img[0];
	}

	friend T* data(image2D& img) noexcept
	{
		return &img.img[0];
	}

	friend auto cbegin(const image2D& img) noexcept
	{
		return data(img);
	}

	friend auto begin(const image2D& img) noexcept
	{
		return data(img);
	}

	friend auto begin(image2D& img) noexcept
	{
		return data(img);
	}

	friend auto cend(const image2D& img) noexcept
	{
		return begin(img) + size(img.width, img.height);
	}

	friend auto end(const image2D& img) noexcept
	{
		return cend(img);
	}

	friend auto end(image2D& img) noexcept
	{
		return data(img) + size(img.width, img.height);
	}

	image2D(std::size_t width, std::size_t height)
		: img(alloc(width, height)),
		  width(width),
		  height(height)
	{
	}

	image2D(const image2D& s)
		: img(alloc(s.width, s.height)),
		  width(s.width),
		  height(s.height)
	{
		std::copy(begin(s), end(s), &img[0]);
	}

	image2D(image2D&& s) = default;

	image2D& operator =(const image2D& s)
	{
		width = s.width;
		height = s.height;
		auto buffer = alloc(width, height);
		std::copy(begin(s), end(s), &buffer[0]);
		img = move(buffer);
		return *this;
	}

	image2D& operator =(image2D&& s) = default;

	T& operator ()(std::size_t x, std::size_t y) const noexcept { return img[y * width + x]; }
	T& operator ()(std::size_t x, std::size_t y) noexcept { return img[y * width + x]; }
};



//std::uint32_t log2(std::uint32_t x)
//{
//	// based on http://aggregate.org/MAGIC/
//	x |= (x >> 1);
//	x |= (x >> 2);
//	x |= (x >> 4);
//	x |= (x >> 8);
//	x |= (x >> 16);
//
//	x -= ((x >> 1) & 0x55555555U);
//	x = (((x >> 2) & 0x33333333U) + (x & 0x33333333U));
//	x = (((x >> 4) + x) & 0x0F0F0F0FU);
//	x += (x >> 8);
//	x += (x >> 16);
//	return (x & 0x0000003FU) - 1;
//}

template <typename T>
class image2DMipmap
{
private:
	std::unique_ptr<T[]> img;

	std::size_t width;
	std::size_t height;
	std::size_t levels;

	static constexpr std::size_t size(std::size_t width, std::size_t height, std::size_t levels)
	{
		return width * height * 2;
	}

	static auto alloc(std::size_t width, std::size_t height, std::size_t levels)
	{
		return std::unique_ptr<T[]> { new T[size(width, height, levels)] };
	}

public:
	image2DMipmap(std::size_t width, std::size_t height, std::size_t levels = 1U)
		: img(alloc(width, height, levels)),
		  width(width),
		  height(height),
		  levels(levels)
	{
	}

	image2DMipmap(const image2DMipmap& s)
		: img(alloc(s.width, s.height, s.levels)),
		  width(s.width),
		  height(s.height),
		  levels(s.levels)
	{
		std::copy(&s.img[0], &s.img[0] + size(width, height, levels), &img[0]);
	}

	image2DMipmap(image2DMipmap&& s) = default;


	image2DMipmap& operator =(const image2DMipmap& s)
	{
		width = s.width;
		height = s.height;
		levels = s.levels;
		auto buffer = alloc(width, height, levels);
		std::copy(&s.img[0], &s.img[0] + size(width, height, levels), &buffer[0]);
		img = move(buffer);
		return *this;
	}

	image2DMipmap& operator =(image2DMipmap&& s) = default;


	T& operator ()(std::size_t x, std::size_t y) const noexcept { return img[y * width + x]; }
	T& operator ()(std::size_t x, std::size_t y) noexcept { return img[y * width + x]; }


	friend std::size_t width(const image2DMipmap& img) noexcept
	{
		return img.width;
	}

	friend std::size_t height(const image2DMipmap& img) noexcept
	{
		return img.height;
	}

	friend std::size_t levels(const image2DMipmap& img) noexcept
	{
		return img.levels;
	}

	friend const T* data(const image2DMipmap& img) noexcept
	{
		return &img.img[0];
	}

	friend T* data(image2DMipmap& img) noexcept
	{
		return &img.img[0];
	}
};

#endif  // INCLUDED_IMAGE
