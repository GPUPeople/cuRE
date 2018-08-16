


#ifndef INCLUDED_WIC
#define INCLUDED_WIC

#pragma once

#include <wincodec.h>

#include <COM/unique_ptr.h>


namespace WIC
{
	IWICImagingFactory* getFactory();

	COM::unique_ptr<IWICStream> createStream(IWICImagingFactory* factory);

	COM::unique_ptr<IWICBitmapEncoder> createEncoder(IWICImagingFactory* factory, REFGUID container_format);

	COM::unique_ptr<IWICBitmapFrameEncode> createEncoderFrame(IWICBitmapEncoder* encoder);

	COM::unique_ptr<IWICBitmapDecoder> createDecoder(IWICImagingFactory* factory, REFGUID container_format);
	COM::unique_ptr<IWICBitmapDecoder> createDecoder(IWICImagingFactory* factory, REFGUID container_format, IStream* stream, WICDecodeOptions options = WICDecodeMetadataCacheOnLoad);
	COM::unique_ptr<IWICBitmapDecoder> createDecoder(IWICImagingFactory* factory, IStream* stream);

	COM::unique_ptr<IWICBitmapFrameDecode> createDecoderFrame(IWICBitmapDecoder* decoder, UINT index = 0U);

	COM::unique_ptr<IWICBitmap> createBitmap(IWICImagingFactory* factory, const GUID& format, UINT width, UINT height, BYTE* data, UINT pitch);

	COM::unique_ptr<IWICFormatConverter> createConverter(IWICImagingFactory* factory);
}

#endif  // INCLUDED_WIC
