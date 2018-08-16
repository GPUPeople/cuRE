


#include <iostream>
#include <fstream>
#include <iomanip>

#include "Config.h"


namespace
{
	std::vector<char> readFile(std::istream& file)
	{
		file.seekg(0, std::ios::end);
		size_t size = static_cast<size_t>(file.tellg());
		file.seekg(0);
		std::vector<char> buffer(size);
		file.read(&buffer[0], size);
		return move(buffer);
	}


	class Log : public virtual ConfigFile::Log
	{
	public:
		void warning(const char* message, const char* file, size_t line, ptrdiff_t column)
		{
			std::cout << file << '(' << line << ',' << column << "): warning: " << message << std::endl;
		}

		void warning(const std::string& message, const char* file, size_t line, ptrdiff_t column)
		{
			std::cout << file << '(' << line << ',' << column << "): warning: " << message << std::endl;
		}

		void error(const char* message, const char* file, size_t line, ptrdiff_t column)
		{
			std::cout << file << '(' << line << ',' << column << "): error: " << message << std::endl;
		}

		void error(const std::string& message, const char* file, size_t line, ptrdiff_t column)
		{
			std::cout << file << '(' << line << ',' << column << "): error: " << message << std::endl;
		}
	};

	std::ostream& writeTuple(std::ostream& stream, const std::vector<std::string>& tuple)
	{
		stream << '(';
		auto v = begin(tuple);
		if (v != end(tuple))
		{
			stream << '"' << *v << '"';
			while (++v != end(tuple))
				stream << ", \"" << *v << '"';
		}
		stream << ')';
		return stream;
	}
}


Config::Config(ConfigFile::Stream& stream)
{
	ConfigFile::parse(stream, *this);
}

Config::Config(Config&& c)
	: string_values(std::move(c.string_values)),
	  int_values(std::move(c.int_values)),
	  float_values(std::move(c.float_values)),
	  configs(std::move(c.configs))
{
}

Config& Config::operator =(Config&& c)
{
	string_values = std::move(c.string_values);
	int_values = std::move(c.int_values);
	float_values = std::move(c.float_values);
	configs = std::move(c.configs);
	return *this;
}

void Config::save(std::ostream& file, const char* indentation) const
{
	for (auto&& v : string_values)
		file << indentation << v.first << " = \"" << v.second << "\"\n";

	for (auto&& v : int_values)
		file << indentation << v.first << " = " << v.second << "\n";

	for (auto&& v : float_values)
		file << indentation << v.first << " = " << std::showpoint << v.second << "\n";

	for (auto&& v : tuples)
	{
		file << indentation << v.first << " = ";
		writeTuple(file, v.second);
		file << "\n";
	}

	for (auto&& v : configs)
	{
		file << indentation << v.first << " = {\n";
		v.second.save(file, (std::string(indentation) + '\t').c_str());
		file << indentation << "}\n";
	}
}


void Config::addString(const std::string& key, const std::string& value)
{
	saveString(key.c_str(), value);
}

void Config::addInt(const std::string& key, int value)
{
	saveInt(key.c_str(), value);
}

void Config::addFloat(const std::string& key, float value)
{
	saveFloat(key.c_str(), value);
}

void Config::addTuple(const std::string& key, std::vector<std::string> value)
{
	saveTuple(key.c_str(), std::move(value));
}

ConfigFile::ParserCallback& Config::addConfig(const std::string& key)
{
	return loadConfig(key.c_str());
}


const char* Config::loadString(const char* key, const char* default_value) const
{
	auto f = string_values.find(key);

	if (f != end(string_values))
		return f->second.c_str();
	return default_value;
}

int Config::loadInt(const char* key, int default_value) const
{
	auto f = int_values.find(key);

	if (f != end(int_values))
		return f->second;
	return default_value;
}

float Config::loadFloat(const char* key, float default_value) const
{
	auto f = float_values.find(key);

	if (f != end(float_values))
		return f->second;
	return default_value;
}

std::vector<std::string> Config::loadTuple(const char* key, const std::initializer_list<const char*>& default_value) const
{
	auto f = tuples.find(key);

	if (f != end(tuples))
		return f->second;

	return {begin(default_value), end(default_value)};
}

const Config& Config::loadConfig(const char* key) const
{
	return configs[key];
}

Config& Config::loadConfig(const char* key)
{
	return configs[key];
}


void Config::saveString(const char* key, const char* value)
{
	string_values[key] = value;
}

void Config::saveString(const char* key, const std::string& value)
{
	string_values[key] = value;
}

void Config::saveString(const char* key, std::string&& value)
{
	string_values[key] = std::move(value);
}

void Config::saveInt(const char* key, int value)
{
	int_values[key] = value;
}

void Config::saveFloat(const char* key, float value)
{
	float_values[key] = value;
}

void Config::saveTuple(const char* key, std::vector<std::string> value)
{
	tuples[key] = std::move(value);
}


Config loadConfig(const char* filename)
{
	std::ifstream file(filename, std::ios::in | std::ios::binary);

	if (file)
	{
		std::vector<char> buffer = readFile(file);

		Log log;
		ConfigFile::Stream stream(&buffer[0], &buffer[0] + buffer.size(), filename, log);

		return Config(stream);
	}

	return Config();
}

void save(const Config& config, const char* filename)
{
	std::ofstream file(filename);
	config.save(file);
}
