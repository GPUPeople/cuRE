


#include <stdexcept>
#include <memory>
#include <stack>
#include <fstream>
#include <iostream>
#include <iomanip>

#include <core/utils/memory>

#include <configfile.h>

#include <configfile/DefaultLog.h>


namespace
{
	class ConfigBuilder : private virtual configfile::ParserCallback
	{
		std::stack<config::Database*> config;

		void addString(const std::string& key, const std::string& value) override
		{
			config.top()->storeString(key.c_str(), value);
		}

		void addInt(const std::string& key, int value) override
		{
			config.top()->storeInt(key.c_str(), value);
		}

		void addFloat(const std::string& key, float value) override
		{
			config.top()->storeFloat(key.c_str(), value);
		}

		void addTuple(const std::string& key, std::vector<std::string> value) override
		{
			throw std::runtime_error("sorry, tuples are currently not supported :(");
		}

		ParserCallback* enterNode(const std::string& key) override
		{
			config.push(&config.top()->openNode(key.c_str()));
			return this;
		}

		void leaveNode() override
		{
			config.pop();
		}

	public:
		ConfigBuilder(config::Database& config)
			: config({ &config })
		{
		}

		configfile::Stream& consume(configfile::Stream& stream)
		{
			return parse(*this, stream);
		}
	};


	class ConfigWriter : public virtual config::Visitor
	{
		std::ostream& file;
		int level = 0;

		std::ostream& indent(std::ostream& file)
		{
			for (int i = 0; i < level; ++i)
				file.put('\t');
			return file;
		}

	public:
		ConfigWriter(std::ostream& file)
			: file(file)
		{
		}

		void visitString(const char* key, const char* value) override
		{
			indent(file) << key << " = \"" << value << "\"\n";
		}

		void visitInt(const char* key, int value) override
		{
			indent(file) << key << " = " << value << '\n';
		}

		void visitFloat(const char* key, float value) override
		{
			indent(file) << key << " = " << std::showpoint << value << '\n';
		}

		Visitor* visitNode(const char* key, const config::Database& node) override
		{
			indent(file) << key << " = {\n";
			++level;
			return this;
		}

		void leaveNode() override
		{
			--level;
			indent(file) << "}\n";
		}
	};
}

namespace configfile
{
	void DefaultLog::warning(const char* message, const char* file, size_t line, ptrdiff_t column)
	{
		std::cerr << file << '(' << line << ',' << column << "): warning: " << message << std::endl;
	}

	void DefaultLog::warning(const std::string& message, const char* file, size_t line, ptrdiff_t column)
	{
		std::cerr << file << '(' << line << ',' << column << "): warning: " << message << std::endl;
	}

	void DefaultLog::error(const char* message, const char* file, size_t line, ptrdiff_t column)
	{
		++num_errors;
		std::cerr << file << '(' << line << ',' << column << "): error: " << message << std::endl;
	}

	void DefaultLog::error(const std::string& message, const char* file, size_t line, ptrdiff_t column)
	{
		++num_errors;
		std::cerr << file << '(' << line << ',' << column << "): error: " << message << std::endl;
	}

	void DefaultLog::throwErrors() const
	{
		if (num_errors)
		{
			std::cerr << num_errors << " errors reading config file" << std::endl;
			throw read_error();
		}
	}


	const char* read_error::what() const noexcept
	{
		return "failed to read config file";
	}


	ParserCallback& read(ParserCallback& parser, const char* begin, const char* end, const char* filename, Log& log)
	{
		Stream stream(begin, end, filename, log);
		parse(parser, stream);
		return parser;
	}

	ParserCallback& read(ParserCallback& parser, const char* begin, const char* end, const char* filename)
	{
		DefaultLog log;
		auto&& ret = read(parser, begin, end, filename, log);
		log.throwErrors();
		return ret;
	}

	std::istream& read(ParserCallback& parser, std::istream& file, const char* filename, Log& log)
	{
		file.seekg(0, std::ios::end);
		size_t size = static_cast<size_t>(file.tellg());
		file.seekg(0);

		if (!file)
			throw read_error();

		auto buffer = core::make_unique_default<char[]>(size);
		file.read(&buffer[0], size);

		if (!file)
			throw read_error();

		read(parser, &buffer[0], &buffer[0] + size, filename, log);

		return file;
	}
	
	std::istream& read(ParserCallback& parser, std::istream& file, const char* filename)
	{
		DefaultLog log;
		auto&& ret = read(parser, file, filename, log);
		log.throwErrors();
		return ret;
	}

	ParserCallback& read(ParserCallback& parser, const char* filename, Log& log)
	{
		std::ifstream file(filename, std::ios::binary);
		read(parser, file, filename, log);
		return parser;
	}

	ParserCallback& read(ParserCallback& parser, const char* filename)
	{
		DefaultLog log;
		auto&& ret = read(parser, filename, log);
		log.throwErrors();
		return ret;
	}


	config::Database& read(config::Database& config, const char* begin, const char* end, const char* filename, Log& log)
	{
		Stream stream(begin, end, filename, log);

		ConfigBuilder builder(config);
		builder.consume(stream);

		return config;
	}

	config::Database& read(config::Database& config, const char* begin, const char* end, const char* filename)
	{
		DefaultLog log;
		auto&& ret = read(config, begin, end, filename, log);
		log.throwErrors();
		return ret;
	}

	std::istream& read(config::Database& config, std::istream& file, const char* filename, Log& log)
	{
		file.seekg(0, std::ios::end);
		size_t size = static_cast<size_t>(file.tellg());
		file.seekg(0);

		if (!file)
			throw read_error();

		auto buffer = core::make_unique_default<char[]>(size);
		file.read(&buffer[0], size);

		if (!file)
			throw read_error();

		read(config, &buffer[0], &buffer[0] + size, filename, log);

		return file;
	}

	std::istream& read(config::Database& config, std::istream& file, const char* filename)
	{
		DefaultLog log;
		auto&& ret = read(config, file, filename, log);
		log.throwErrors();
		return ret;
	}

	config::Database& read(config::Database& config, const char* filename, Log& log)
	{
		std::ifstream file(filename, std::ios::binary);
		read(config, file, filename, log);
		return config;
	}

	config::Database& read(config::Database& config, const char* filename)
	{
		DefaultLog log;
		auto&& ret = read(config, filename, log);
		log.throwErrors();
		return ret;
	}


	std::ostream& write(std::ostream& file, const config::Database& config)
	{
		ConfigWriter writer(file);
		config.traverse(writer);
		return file;
	}

	void write(const char* filename, const config::Database& config)
	{
		std::ofstream file(filename, std::ios::binary);
		write(file, config);
	}
}
