


#include <cassert>
#include <limits>
#include <memory>

#include <win32/error.h>

#include <win32/unicode.h>


namespace
{
	std::basic_string<WCHAR> utf8_to_utf16(const char* input, int input_length)
	{
		int output_length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, input, input_length, nullptr, 0);

		if (output_length <= 0)
			Win32::throw_last_error();

		auto output = std::unique_ptr<WCHAR[]> { new WCHAR[output_length] };

		if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, input, input_length, &output[0], output_length) <= 0)
			Win32::throw_last_error();

		return { &output[0], static_cast<std::size_t>(output_length) };
	}

	std::basic_string<WCHAR> utf8_to_utf16(const char* input)
	{
		int output_length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, input, -1, nullptr, 0);

		if (output_length <= 0)
			Win32::throw_last_error();

		auto output = std::unique_ptr<WCHAR[]> { new WCHAR[output_length] };

		if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, input, -1, &output[0], output_length) <= 0)
			Win32::throw_last_error();

		return { &output[0], static_cast<std::size_t>(output_length - 1) };
	}

	std::string utf16_to_utf8(const WCHAR* input, int input_length)
	{
		int output_length = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, input, input_length, nullptr, 0, nullptr, nullptr);

		if (output_length <= 0)
			Win32::throw_last_error();

		auto output = std::unique_ptr<char[]> { new char[output_length] };

		if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, input, input_length, &output[0], output_length, nullptr, nullptr) <= 0)
			Win32::throw_last_error();

		return { &output[0], static_cast<std::size_t>(output_length) };
	}

	std::string utf16_to_utf8(const WCHAR* input)
	{
		int output_length = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, input, -1, nullptr, 0, nullptr, nullptr);

		if (output_length <= 0)
			Win32::throw_last_error();

		auto output = std::unique_ptr<char[]> { new char[output_length] };

		if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, input, -1, &output[0], output_length, nullptr, nullptr) <= 0)
			Win32::throw_last_error();

		return { &output[0], static_cast<std::size_t>(output_length - 1) };
	}
}

namespace Win32
{
	std::basic_string<WCHAR> widen(const CHAR* string, size_t length)
	{
		assert(length < static_cast<size_t>(std::numeric_limits<int>::max()));
		return utf8_to_utf16(string, static_cast<int>(length));
	}

	std::basic_string<WCHAR> widen(const CHAR* string)
	{
		return utf8_to_utf16(string);
	}

	std::basic_string<WCHAR> widen(const std::string& string)
	{
		return widen(&string[0], string.length());
	}


	std::string narrow(const WCHAR* string, size_t length)
	{
		assert(length < static_cast<size_t>(std::numeric_limits<int>::max()));
		return utf16_to_utf8(string, static_cast<int>(length));
	}

	std::string narrow(const WCHAR* string)
	{
		return utf16_to_utf8(string);
	}

	std::string narrow(const std::basic_string<WCHAR>& string)
	{
		return narrow(&string[0], string.length());
	}
}
