


#include <win32/unicode.h>
#include <COM/error.h>

#include "wic.h"

#include "png.h"


namespace
{
	COM::unique_ptr<IWICBitmapDecoder> createDecoder(IWICImagingFactory* factory, IStream* stream)
	{
		return WIC::createDecoder(factory, GUID_ContainerFormatPng, stream);
	}

	COM::unique_ptr<IWICBitmapEncoder> createEncoder(IWICImagingFactory* factory)
	{
		return WIC::createEncoder(factory, GUID_ContainerFormatPng);
	}

	std::tuple<std::size_t, std::size_t> readSize(IStream* stream, IWICBitmapDecoder* decoder)
	{
		COM::unique_ptr<IWICBitmapFrameDecode> frame = WIC::createDecoderFrame(decoder);

		UINT width;
		UINT height;
		COM::throw_error(frame->GetSize(&width, &height));

		return std::make_tuple(width, height);
	}

	image2D<RGBA8> loadRGBA8(IStream* stream, IWICImagingFactory* factory)
	{
		COM::unique_ptr<IWICBitmapDecoder> decoder = createDecoder(factory, stream);

		COM::unique_ptr<IWICBitmapFrameDecode> frame = WIC::createDecoderFrame(decoder);

		UINT width;
		UINT height;
		COM::throw_error(frame->GetSize(&width, &height));

		image2D<RGBA8> image(width, height);

		WICPixelFormatGUID pixel_format;
		COM::throw_error(frame->GetPixelFormat(&pixel_format));

		if (IsEqualGUID(pixel_format, GUID_WICPixelFormat32bppRGBA))
		{
			COM::throw_error(frame->CopyPixels(nullptr, width * 4, width * height * 4, reinterpret_cast<BYTE*>(data(image))));
		}
		else
		{
			COM::unique_ptr<IWICFormatConverter> format_converter = WIC::createConverter(factory);

			COM::throw_error(format_converter->Initialize(frame, GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom));

			COM::throw_error(format_converter->CopyPixels(nullptr, width * 4, width * height * 4, reinterpret_cast<BYTE*>(data(image))));
		}

		return image;
	}

	void saveRGBA8(IStream* stream, int width, int height, const void* data, int pitch, IWICImagingFactory* factory)
	{
		COM::unique_ptr<IWICBitmapEncoder> encoder = createEncoder(factory);

		COM::throw_error(encoder->Initialize(stream, WICBitmapEncoderNoCache));

		COM::unique_ptr<IWICBitmapFrameEncode> frame = WIC::createEncoderFrame(encoder);

		COM::throw_error(frame->Initialize(nullptr));

		COM::throw_error(frame->SetSize(width, height));

		WICPixelFormatGUID pixel_format = GUID_WICPixelFormat32bppRGBA;
		COM::throw_error(frame->SetPixelFormat(&pixel_format));

		COM::unique_ptr<IWICBitmap> bitmap = WIC::createBitmap(factory, GUID_WICPixelFormat32bppRGBA, width, height, static_cast<BYTE*>(const_cast<void*>(data)), pitch);

		if (IsEqualGUID(pixel_format, GUID_WICPixelFormat32bppRGBA))
		{
			COM::throw_error(frame->WriteSource(bitmap, nullptr));
		}
		else
		{
			COM::unique_ptr<IWICFormatConverter> converter = WIC::createConverter(factory);

			COM::throw_error(converter->Initialize(bitmap, pixel_format, WICBitmapDitherTypeNone, nullptr, 0.0f, WICBitmapPaletteTypeCustom));
			COM::throw_error(frame->WriteSource(converter, nullptr));
		}

		COM::throw_error(frame->Commit());
		COM::throw_error(encoder->Commit());
	}
}

namespace PNG
{
	std::tuple<std::size_t, std::size_t> readSize(const wchar_t* filename)
	{
		auto factory = WIC::getFactory();
		auto stream = WIC::createStream(factory);
		stream->InitializeFromFilename(filename, GENERIC_READ);
		auto decoder = createDecoder(factory, stream);
		return ::readSize(stream, decoder);
	}

	std::tuple<std::size_t, std::size_t> readSize(const char* filename)
	{
		return readSize(Win32::widen(filename).c_str());
	}

	image2D<RGBA8> loadRGBA8(const void* data, size_t data_size)
	{
		auto factory = WIC::getFactory();
		auto stream = WIC::createStream(factory);
		stream->InitializeFromMemory(static_cast<BYTE*>(const_cast<void*>(data)), static_cast<DWORD>(data_size));
		return ::loadRGBA8(stream, factory);
	}

	image2D<RGBA8> loadRGBA8(const wchar_t* filename)
	{
		auto factory = WIC::getFactory();
		auto stream = WIC::createStream(factory);
		stream->InitializeFromFilename(filename, GENERIC_READ);
		return ::loadRGBA8(stream, factory);
	}

	image2D<RGBA8> loadRGBA8(const char* filename)
	{
		return loadRGBA8(Win32::widen(filename).c_str());
	}

	void saveRGBA8(const wchar_t* filename, const image2D<RGBA8>& image)
	{
		auto factory = WIC::getFactory();
		auto stream = WIC::createStream(factory);
		stream->InitializeFromFilename(filename, GENERIC_WRITE);
		::saveRGBA8(stream, static_cast<int>(width(image)), static_cast<int>(height(image)), data(image), static_cast<int>(width(image)) * 4, factory);
	}

	void saveRGBA8(const char* filename, const image2D<RGBA8>& image)
	{
		return saveRGBA8(Win32::widen(filename).c_str(), image);
	}
}
