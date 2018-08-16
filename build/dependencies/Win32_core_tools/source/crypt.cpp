


#include <win32/error.h>
#include <win32/crypt.h>


namespace Win32
{
	namespace Crypt
	{
		unique_provider acquireContext(const wchar_t* container, const wchar_t* provider, DWORD provider_type, DWORD flags)
		{
			HCRYPTPROV prov;
			if (!CryptAcquireContextW(&prov, container, provider, provider_type, flags))
				Win32::throw_last_error();
			return unique_provider(prov);
		}

		unique_hash createHash(HCRYPTPROV provider, ALG_ID algorithm, HCRYPTKEY key, DWORD flags)
		{
			HCRYPTHASH hash;
			if (!CryptCreateHash(provider, algorithm, key, flags, &hash))
				Win32::throw_last_error();
			return unique_hash(hash);
		}
	}
}
