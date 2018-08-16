


#include <cstdint>
#include <string>
#include <stdexcept>
#include <iostream>

#include "binary.h"


namespace
{
	class memory_istreambuf : public std::basic_streambuf < char >
	{
	protected:
		pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which)
		{
			if (which & std::ios_base::out || dir != std::ios_base::cur)
				return pos_type(off_type(-1));

			setg(eback(), gptr() + off, egptr());
			return gptr() - eback();
		}

	public:
		memory_istreambuf(char* begin, char* end)
		{
			setg(begin, begin, end);
		}

		const char* current() const
		{
			return gptr();
		}

		char* current()
		{
			return gptr();
		}
	};

	template <typename T>
	T read(const char* address)
	{
		return *reinterpret_cast<const T*>(address);
	}

	void checkHeader(const char* image)
	{
		if (read<std::uint8_t>(image + 0x0U) != '\x7F' ||
		    read<std::uint8_t>(image + 0x1U) != 'E' ||
		    read<std::uint8_t>(image + 0x2U) != 'L' ||
		    read<std::uint8_t>(image + 0x3U) != 'F')
			throw std::runtime_error("not a valid ELF binary");

		auto e_class = read<std::uint8_t>(image + 0x4U);
		auto e_data = read<std::uint8_t>(image + 0x5U);
		//auto e_version = read<std::uint8_t>(image + 0x6U);
		auto e_osabi = read<std::uint8_t>(image + 0x7U);
		auto e_abi_version = read<std::uint8_t>(image + 0x8U);

		if (e_class != 2)
			throw std::runtime_error("only 64-bit binaries are supported atm");
		if (e_data != 1)
			throw std::runtime_error("only little-endian binaries are supported atm");

		auto e_type = read<std::uint16_t>(image + 0x10U);
		auto e_machine = read<std::uint16_t>(image + 0x12U);

		if (e_machine != 190)
			throw std::runtime_error("not a CUDA binary");
	}
}

namespace CU
{
	std::tuple<int, int> readComputeCapability(const char* image)
	{
		checkHeader(image);

		auto e_flags = read<std::uint32_t>(image + 0x30U);

		unsigned int cc = (e_flags & 0xFFFF0000U) >> 16;

		return std::make_tuple(cc / 10, cc % 10);
	}

	std::vector<const char*> readSymbols(const char* image)
	{
		checkHeader(image);

		std::vector<const char*> symbols;

		//auto e_version = read<std::uint32_t>(image + 0x14U);
		//auto e_entry = read<std::uint64_t>(image + 0x18U);
		//auto e_phoff = read<std::uint64_t>(image + 0x20U);
		auto e_shoff = read<std::uint64_t>(image + 0x28U);
		//auto e_flags = read<std::uint32_t>(image + 0x30U);
		//auto e_ehsize = read<std::uint16_t>(image + 0x34U);
		//auto e_phentsize = read<std::uint16_t>(image + 0x36U);
		//auto e_phnum = read<std::uint16_t>(image + 0x38U);
		auto e_shentsize = read<std::uint16_t>(image + 0x3AU);
		auto e_shnum = read<std::uint16_t>(image + 0x3CU);
		auto e_shstrndx = read<std::uint16_t>(image + 0x3EU);

		auto sh_string_table_offset = read<std::uint64_t>(image + (e_shoff + e_shstrndx * e_shentsize) + 24);

		const char* sh_string_table = image + sh_string_table_offset;

		for (int i = 0; i < e_shnum; ++i)
		{
			const char* psh = image + (e_shoff + i * e_shentsize);

			auto sh_name = read<std::uint32_t>(psh);
			auto sh_type = read<std::uint32_t>(psh + 4);

			if (sh_type == 2)
			{
				// symbol table

				//auto sh_flags = read<std::uint64_t>(psh + 8);
				//auto sh_addr = read<std::uint64_t>(psh + 16);
				auto sh_offset = read<std::uint64_t>(psh + 24);
				auto sh_size = read<std::uint64_t>(psh + 32);
				auto sh_link = read<std::uint32_t>(psh + 40);
				//auto sh_info = read<std::uint32_t>(psh + 44);
				//auto sh_addralign = read<std::uint64_t>(psh + 48);
				auto sh_entsize = read<std::uint64_t>(psh + 56);

				const char* section_name = sh_string_table + sh_name;

				auto string_table_offset = read<std::uint64_t>(image + (e_shoff + sh_link * e_shentsize) + 24);
				const char* string_table = image + string_table_offset;

				auto num_entries = sh_size / sh_entsize;

				for (const char* psym = image + sh_offset; psym < image + sh_offset + sh_size; psym += sh_entsize)
				{
					auto st_name = read<std::uint32_t>(psym);
					auto st_info = read<std::uint8_t>(psym + 4);
					//auto st_other = read<std::uint8_t>(psym + 5);
					//auto st_shndx = read<std::uint16_t>(psym + 6);
					//auto st_addr = read<std::uint64_t>(psym + 8);
					auto st_size = read<std::uint64_t>(psym + 16);

					const char* symbol_name = string_table + st_name;

					if ((st_info & 0xFU) == 3)
					{
						symbols.push_back(symbol_name);
					}
				}
			}
		}

		return symbols;
	}
}
