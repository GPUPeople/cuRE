


#ifndef INCLUDED_WIN32_CRYPT
#define INCLUDED_WIN32_CRYPT

#pragma once

#include <win32/platform.h>
#include <win32/unique_handle.h>

#include <wincrypt.h>


namespace Win32
{
	namespace Crypt
	{
		struct CryptReleaseContextDeleter
		{
			void operator ()(HCRYPTPROV provider) const
			{
				CryptReleaseContext(provider, 0U);
			}
		};

		using unique_provider = Win32::unique_handle<HCRYPTPROV, 0U, CryptReleaseContextDeleter>;

		unique_provider acquireContext(const wchar_t* container, const wchar_t* provider, DWORD provider_type, DWORD flags);


		struct CryptDestroyHashDeleter
		{
			void operator ()(HCRYPTHASH hash) const
			{
				CryptDestroyHash(hash);
			}
		};

		using unique_hash = Win32::unique_handle<HCRYPTHASH, 0U, CryptDestroyHashDeleter>;

		unique_hash createHash(HCRYPTPROV provider, ALG_ID algorithm, HCRYPTKEY key = 0, DWORD flags = 0U);
	}
}

#endif  // INCLUDED_WIN32_CRYPT
