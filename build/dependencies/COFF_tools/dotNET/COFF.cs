using System;
using System.Collections.Generic;
using System.Text;
using System.IO;


namespace COFF
{
	public enum MachineType : uint
	{
		UNKNOWN   = 0x0U,
		AM33      = 0x1D3U,
		AMD64     = 0x8664U,
		ARM       = 0x1C0U,
		ARMV7     = 0x1C4U,
		EBC       = 0xEBCU,
		I386      = 0x14CU,
		IA64      = 0x200U,
		M32R      = 0x9041U,
		MIPS16    = 0x266U,
		MIPSFPU   = 0x366U,
		MIPSFPU16 = 0x466U,
		POWERPC   = 0x1F0U,
		POWERPCFP = 0x1F1U,
		R4000     = 0x166U,
		SH3       = 0x1A2U,
		SH3DSP    = 0x1A3U,
		SH4       = 0x1A6U,
		SH5       = 0x1A8U,
		THUMB     = 0x1C2U,
		WCEMIPSV2 = 0x169U
	}

	[Flags]
	public enum SectionCharacteristics : uint
	{
		TYPE_NO_PAD            = 0x00000008U,
		CNT_CODE               = 0x00000020U,
		CNT_INITIALIZED_DATA   = 0x00000040U,
		CNT_UNINITIALIZED_DATA = 0x00000080U,
		LNK_OTHER              = 0x00000100U,
		LNK_INFO               = 0x00000200U,
		LNK_REMOVE             = 0x00000800U,
		LNK_COMDAT             = 0x00001000U,
		GPREL                  = 0x00008000U,
		MEM_PURGEABLE          = 0x00020000U,
		MEM_16BIT              = 0x00020000U,
		MEM_LOCKED             = 0x00040000U,
		MEM_PRELOAD            = 0x00080000U,
		ALIGN_1BYTES           = 0x00100000U,
		ALIGN_2BYTES           = 0x00200000U,
		ALIGN_4BYTES           = 0x00300000U,
		ALIGN_8BYTES           = 0x00400000U,
		ALIGN_16BYTES          = 0x00500000U,
		ALIGN_32BYTES          = 0x00600000U,
		ALIGN_64BYTES          = 0x00700000U,
		ALIGN_128BYTES         = 0x00800000U,
		ALIGN_256BYTES         = 0x00900000U,
		ALIGN_512BYTES         = 0x00A00000U,
		ALIGN_1024BYTES        = 0x00B00000U,
		ALIGN_2048BYTES        = 0x00C00000U,
		ALIGN_4096BYTES        = 0x00D00000U,
		ALIGN_8192BYTES        = 0x00E00000U,
		LNK_NRELOC_OVFL        = 0x01000000U,
		MEM_DISCARDABLE        = 0x02000000U,
		MEM_NOT_CACHED         = 0x04000000U,
		MEM_NOT_PAGED          = 0x08000000U,
		MEM_SHARED             = 0x10000000U,
		MEM_EXECUTE            = 0x20000000U,
		MEM_READ               = 0x40000000U,
		MEM_WRITE              = 0x80000000U
	}

	[Flags]
	public enum SymbolType : uint
	{
		NULL   =  0U,
		VOID   =  1U,
		CHAR   =  2U,
		SHORT  =  3U,
		INT    =  4U,
		LONG   =  5U,
		FLOAT  =  6U,
		DOUBLE =  7U,
		STRUCT =  8U,
		UNION  =  9U,
		ENUM   = 10U,
		MOE    = 11U,
		BYTE   = 12U,
		WORD   = 13U,
		UINT   = 14U,
		DWORD  = 15U,

		POINTER  = 1U << 8,
		FUNCTION = 2U << 8,
		ARRAY    = 3U << 8
	}

	public enum StorageClass : uint
	{
		END_OF_FUNCTION  = 0xFFU,
		NULL             =    0U,
		AUTOMATIC        =    1U,
		EXTERNAL         =    2U,
		STATIC           =    3U,
		REGISTER         =    4U,
		EXTERNAL_DEF     =    5U,
		LABEL            =    6U,
		UNDEFINED_LABEL  =    7U,
		MEMBER_OF_STRUCT =    8U,
		ARGUMENT         =    9U,
		STRUCT_TAG       =   10U,
		MEMBER_OF_UNION  =   11U,
		UNION_TAG        =   12U,
		TYPE_DEFINITION  =   13U,
		UNDEFINED_STATIC =   14U,
		ENUM_TAG         =   15U,
		MEMBER_OF_ENUM   =   16U,
		REGISTER_PARAM   =   17U,
		BIT_FIELD        =   18U,
		BLOCK            =  100U,
		FUNCTION         =  101U,
		END_OF_STRUCT    =  102U,
		FILE             =  103U,
		SECTION          =  104U,
		WEAK_EXTERNAL    =  105U,
		CLR_TOKEN        =  107U
	}

	public delegate void SymbolCallback(String name, UInt32 address, SymbolType type, StorageClass storage_class);

	public interface Section
	{
		String Name { get; }
		UInt32 Size { get; }
		SectionCharacteristics Characteristics { get; }
		UInt32 SymbolCount { get; }
		void EnumerateSymbols(SymbolCallback callback);
		void WriteData(Stream stream);
	}

	public class StringTable
	{
		List<Byte[]> string_table;
		UInt32 next_string_offset;

		public StringTable()
		{
			string_table = new List<byte[]>();
			next_string_offset = 4U;
		}

		public UInt32 allocString(String str)
		{
			return allocString(UTF8Encoding.UTF8.GetBytes(str));
		}

		public UInt32 allocString(Byte[] bytes)
		{
			UInt32 offset = next_string_offset;
			string_table.Add(bytes);
			next_string_offset += (UInt32)bytes.Length + 1U;
			return offset;
		}

		public void write(BinaryWriter stream)
		{
			stream.Write((UInt32)next_string_offset);
			foreach (Byte[] str in string_table)
			{
				stream.Write(str);
				stream.Write((Byte)0U);
			}
		}
	}

	public class Object
	{
		struct SectionTableEntry
		{
			public Section section;
			public UInt32 offset;

			public SectionTableEntry(Section section, UInt32 offset)
			{
				this.section = section;
				this.offset = offset;
			}
		}

		UInt32 symbol_count;
		List<SectionTableEntry> section_table;
		UInt32 next_section_offset;

		public Object()
		{
			symbol_count = 0U;
			section_table = new List<SectionTableEntry>();
			next_section_offset = 0U;
		}

		public void add(Section section)
		{
			section_table.Add(new SectionTableEntry(section, next_section_offset));
			next_section_offset = (next_section_offset + section.Size + 3U) & 0xFFFFFFFCU;
			symbol_count += section.SymbolCount;
		}

		static void pad(Stream stream, UInt32 pos)
		{
			pad(stream, pos, 0);
		}

		static void pad(Stream stream, UInt32 pos, Byte value)
		{
			while (stream.Position < pos)
				stream.WriteByte(value);
		}

		static void padN(Stream stream, int count)
		{
			padN(stream, count, 0);
		}

		static void padN(Stream stream, int count, Byte value)
		{
			for (int i = 0; i < count; ++i)
				stream.WriteByte(value);
		}

		public void write(Stream stream)
		{
			using (BinaryWriter obj = new BinaryWriter(stream))
			{
				var string_table = new StringTable();

				UInt32 section_data_offset = 20U + 40U * (UInt32)section_table.Count;
				UInt32 symbol_table_offset = section_data_offset + next_section_offset;

				// COFF Header
				obj.Write((UInt16)MachineType.UNKNOWN);        // Machine
				obj.Write((UInt16)section_table.Count);        // NumberOfSections
				obj.Write((UInt32)(DateTime.Now - new DateTime(1970, 1, 1, 0, 0, 0, 0)).TotalSeconds);  // TimeDateStamp
				obj.Write((UInt32)symbol_table_offset);        // PointerToSymbolTable
				obj.Write((UInt32)symbol_count);               // NumberOfSymbols
				obj.Write((UInt16)0U);                         // SizeOfOptionalHeader
				obj.Write((UInt16)0U);                         // Characteristics

				// COFF Section Table
				foreach (var ste in section_table)
				{
					Byte[] name = UTF8Encoding.UTF8.GetBytes(ste.section.Name);
					if (name.Length > 8)
					{
						UInt32 name_offset = string_table.allocString(name);
						name = ASCIIEncoding.ASCII.GetBytes(String.Format("/{0:D}", name_offset));
					}
					obj.Write(name); padN(stream, 8 - name.Length); // Name

					obj.Write((UInt32)0U);  // VirtualSize
					obj.Write((UInt32)0U);  // VirtualAddress
					obj.Write((UInt32)ste.section.Size);  // SizeOfRawData
					obj.Write((UInt32)(section_data_offset + ste.offset));  // PointerToRawData
					obj.Write((UInt32)0U);  // PointerToRelocations
					obj.Write((UInt32)0U);  // PointerToLinenumbers
					obj.Write((UInt16)0U);  // NumberOfRelocations
					obj.Write((UInt16)0U);  // NumberOfLinenumbers
					obj.Write((UInt32)ste.section.Characteristics);  // Characteristics
				}

				// Section Data
				foreach (var ste in section_table)
				{
					pad(stream, section_data_offset + ste.offset);
					ste.section.WriteData(stream);
				}

				pad(stream, symbol_table_offset);

				// COFF Symbol Table
				for (int i = 0; i < section_table.Count; ++i)
				{
					section_table[i].section.EnumerateSymbols(delegate(String name, UInt32 address, SymbolType type, StorageClass storage_class)
					{
						Byte[] symbol_name = UTF8Encoding.UTF8.GetBytes(name);
						if (symbol_name.Length > 8)
						{
							obj.Write((UInt32)0U);                                     // Zeros
							obj.Write((UInt32)string_table.allocString(symbol_name));  // Name Offset
						}
						else
						{
							obj.Write(symbol_name);
							padN(stream, 8 - symbol_name.Length);  // Name
						}

						obj.Write((UInt32)address);      // Address
						obj.Write((UInt16)(i + 1));      // SectionNumber
						obj.Write((UInt16)type);         // Type
						obj.Write((Byte)storage_class);  // StorageClass
						obj.Write((Byte)0U);             // NumberOfAuxSymbols
					});
				}

				// COFF String Table
				string_table.write(obj);
			}
		}
	}
}
