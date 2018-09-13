


#include <COM/init.h>
#include <COM/error.h>

#include "wic.h"


namespace
{
	COM::unique_ptr<IWICImagingFactory> createWICFactory()
	{
		IWICImagingFactory* factory;
		COM::throw_error(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_IWICImagingFactory, reinterpret_cast<void**>(&factory)));
		return COM::make_unique_ptr(factory);
	}
}

namespace WIC
{
	IWICImagingFactory* getFactory()
	{
		static COM::scope com_init;
		static auto factory = createWICFactory();
		return factory;
	}

	COM::unique_ptr<IWICStream> createStream(IWICImagingFactory* factory)
	{
		IWICStream* stream;
		COM::throw_error(factory->CreateStream(&stream));
		return COM::make_unique_ptr(stream);
	}

	COM::unique_ptr<IWICBitmapEncoder> createEncoder(IWICImagingFactory* factory, REFGUID container_format)
	{
		IWICBitmapEncoder* encoder;
		COM::throw_error(factory->CreateEncoder(container_format, nullptr, &encoder));
		return COM::make_unique_ptr(encoder);
	}

	COM::unique_ptr<IWICBitmapFrameEncode> createEncoderFrame(IWICBitmapEncoder* encoder)
	{
		IWICBitmapFrameEncode* frame;
		COM::throw_error(encoder->CreateNewFrame(&frame, nullptr));
		return COM::make_unique_ptr(frame);
	}

	COM::unique_ptr<IWICBitmapDecoder> createDecoder(IWICImagingFactory* factory, REFGUID container_format)
	{
		IWICBitmapDecoder* decoder;
		COM::throw_error(factory->CreateDecoder(container_format, nullptr, &decoder));
		return COM::make_unique_ptr(decoder);
	}

	COM::unique_ptr<IWICBitmapDecoder> createDecoder(IWICImagingFactory* factory, REFGUID container_format, IStream* stream, WICDecodeOptions options)
	{
		auto decoder = createDecoder(factory, container_format);
		decoder->Initialize(stream, options);
		return decoder;
	}

	COM::unique_ptr<IWICBitmapDecoder> createDecoder(IWICImagingFactory* factory, IStream* stream)
	{
		IWICBitmapDecoder* decoder;
		COM::throw_error(factory->CreateDecoderFromStream(stream, nullptr, WICDecodeMetadataCacheOnDemand, &decoder));
		return COM::make_unique_ptr(decoder);
	}

	COM::unique_ptr<IWICBitmapFrameDecode> createDecoderFrame(IWICBitmapDecoder* decoder, UINT index)
	{
		IWICBitmapFrameDecode* frame;
		COM::throw_error(decoder->GetFrame(index, &frame));
		return COM::make_unique_ptr(frame);
	}

	COM::unique_ptr<IWICBitmap> createBitmap(IWICImagingFactory* factory, const GUID& format, UINT width, UINT height, BYTE* data, UINT pitch)
	{
		IWICBitmap* bitmap;
		COM::throw_error(factory->CreateBitmapFromMemory(width, height, format, pitch, pitch * height, data, &bitmap));
		return COM::make_unique_ptr(bitmap);
	}

	COM::unique_ptr<IWICFormatConverter> createConverter(IWICImagingFactory* factory)
	{
		IWICFormatConverter* converter;
		COM::throw_error(factory->CreateFormatConverter(&converter));
		return COM::make_unique_ptr(converter);
	}
}
