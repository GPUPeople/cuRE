using System;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;
using Microsoft.Build.Framework;
using Microsoft.Build.Utilities;


namespace embedCUDA
{
	class LogListener
	{
		TaskLoggingHelper log;

		public LogListener(TaskLoggingHelper log)
		{
			this.log = log;
		}

		void StdOutCallback(object sender, DataReceivedEventArgs e)
		{
			if (e.Data != null)
				log.LogMessageFromText(e.Data, MessageImportance.High);
		}

		void StdErrCallback(object sender, DataReceivedEventArgs e)
		{
			if (e.Data != null)
				log.LogMessageFromText(e.Data, MessageImportance.High);
		}

		public void Attach(Process process)
		{
			process.OutputDataReceived += StdOutCallback;

			process.ErrorDataReceived += StdErrCallback;

			process.BeginOutputReadLine();
			process.BeginErrorReadLine();
		}
	}
}
