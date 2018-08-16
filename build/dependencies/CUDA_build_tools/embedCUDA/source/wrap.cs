using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using Microsoft.Build.Framework;
using Microsoft.Build.Utilities;


namespace embedCUDA
{
	public class wrap : Task
	{
		[Required]
		public ITaskItem[] Inputs
		{
			get;
			set;
		}

		[Required]
		public String[] SymbolNames
		{
			get;
			set;
		}

		[Required]
		public String[] EndSymbolNames
		{
			get;
			set;
		}

		[Required]
		public String ObjectFile
		{
			get;
			set;
		}

		[Output]
		public ITaskItem Output
		{
			get;
			private set;
		}

		public override bool Execute()
		{
			if (ObjectFile == null || ObjectFile == "")
				return true;

			var binaries = new CUDABinary[Inputs.Length];

			try
			{
				var obj = new COFF.Object();

				for (int i = 0; i < Inputs.Length; ++i)
				{
					binaries[i] = new CUDABinary(Inputs[i].ItemSpec, SymbolNames[i], EndSymbolNames[i]);
					if (binaries[i].SymbolCount == 0U)
						Log.LogWarning("no symbols specified for CUDA binary '{0}'", Inputs[i]);
					obj.add(binaries[i]);
				}

				using (FileStream objfile = File.Create(ObjectFile))
					obj.write(objfile);

				Output = new TaskItem(ObjectFile);
			}
			catch (Exception e)
			{
				Log.LogErrorFromException(e);
				return false;
			}

			return true;
		}
	}
}
