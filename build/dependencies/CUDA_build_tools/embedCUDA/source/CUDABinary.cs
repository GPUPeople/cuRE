using System;
using System.Collections.Generic;
using System.Text;
using System.IO;


namespace embedCUDA
{
	class CUDABinary : COFF.Section
	{
		FileInfo file_info;

		String symbol_name;
		String end_symbol_name;

		public String Name
		{
			get { return ".cuda"; }
		}

		public UInt32 Size
		{
			get { return (UInt32)file_info.Length; }
		}

		public COFF.SectionCharacteristics Characteristics
		{
			get { return COFF.SectionCharacteristics.CNT_INITIALIZED_DATA | COFF.SectionCharacteristics.MEM_READ | COFF.SectionCharacteristics.ALIGN_16BYTES; }
		}

		public UInt32 SymbolCount
		{
			get { return (symbol_name != null ? 1U : 0U) + (end_symbol_name != null ? 1U : 0U); }
		}

		public void EnumerateSymbols(COFF.SymbolCallback callback)
		{
			if (symbol_name != null)
				callback(symbol_name, 0U, COFF.SymbolType.NULL, COFF.StorageClass.EXTERNAL);
			if (end_symbol_name != null)
				callback(end_symbol_name, Size, COFF.SymbolType.NULL, COFF.StorageClass.EXTERNAL);
		}

		public void WriteData(Stream stream)
		{
			using (FileStream file = file_info.Open(FileMode.Open, FileAccess.Read))
				file.CopyTo(stream);
		}

		public CUDABinary(String filename, String symbol_name, String end_symbol_name)
		{
			this.symbol_name = symbol_name;
			this.end_symbol_name = end_symbol_name;
			if (symbol_name == "")
				symbol_name = null;
			if (end_symbol_name == "")
				end_symbol_name = null;
			file_info = new FileInfo(filename);
		}
	}
}
