using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Diagnostics;
using Microsoft.Build.Framework;
using Microsoft.Build.Utilities;


namespace embedCUDA
{
	public class nvlink : Task
	{
		[Required]
		public ITaskItem[] Inputs
		{
			get;
			set;
		}

		[Required]
		public String OutputFile
		{
			get;
			set;
		}

		[Required]
		public String CodeGeneration
		{
			get;
			set;
		}

		[Required]
		public String TargetMachinePlatform
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

		String getGPUArchitecture()
		{
			var parts = CodeGeneration.Split(',');
			return parts[1];
		}

		String buildCmdLine()
		{
			var cmd_line = new StringBuilder();
			cmd_line.Append("-arch ");
			cmd_line.Append(getGPUArchitecture());
			cmd_line.Append(" -m ");
			cmd_line.Append(TargetMachinePlatform);
			cmd_line.Append(" -o \"");
			cmd_line.Append(OutputFile);
			cmd_line.Append('"');

			foreach (var input in Inputs)
			{
				cmd_line.Append(" \"");
				cmd_line.Append(input);
				cmd_line.Append('"');
			}

			return cmd_line.ToString();
		}

		int Link(String cmdline, LogListener log_listener)
		{
			Log.LogMessageFromText("nvlink " + cmdline, MessageImportance.Normal);
			Log.LogCommandLine(MessageImportance.Normal, "nvlink " + cmdline);

			var sinfo = new ProcessStartInfo("nvlink", cmdline);

			sinfo.UseShellExecute = false;
			sinfo.RedirectStandardError = true;
			sinfo.RedirectStandardOutput = true;

			Process process = Process.Start(sinfo);

			log_listener.Attach(process);

			process.WaitForExit();
			int exit_code = process.ExitCode;
			return exit_code;
		}

		public override bool Execute()
		{
			var log = new LogListener(Log);

			bool success = Link(buildCmdLine(), log) == 0;

			Output = new TaskItem(OutputFile);

			return success;
		}
	}
}
