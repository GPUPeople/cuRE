


#ifndef INCLUDED_CONFIG
#define INCLUDED_CONFIG

#pragma once

#include <initializer_list>
#include <iosfwd>
#include <unordered_map>
#include <string>
#include <vector>


#include "ConfigParser.h"


class Config : private virtual ConfigFile::ParserCallback
{
private:
	std::unordered_map<std::string, std::string> string_values;
	std::unordered_map<std::string, int> int_values;
	std::unordered_map<std::string, float> float_values;
	std::unordered_map<std::string, std::vector<std::string>> tuples;
	mutable std::unordered_map<std::string, Config> configs;

	void addString(const std::string& key, const std::string& value);
	void addInt(const std::string& key, int value);
	void addFloat(const std::string& key, float value);
	void addTuple(const std::string& key, std::vector<std::string> value);
	ConfigFile::ParserCallback& addConfig(const std::string& key);

public:
	Config() = default;
	Config(ConfigFile::Stream& stream);

	Config(const Config& c) = default;
	Config(Config&& c);
	Config& operator =(const Config& c) = default;
	Config& operator =(Config&& c);

	const char* loadString(const char* key, const char* default_value) const;
	int loadInt(const char* key, int default_value) const;
	float loadFloat(const char* key, float default_value) const;
	std::vector<std::string> loadTuple(const char* key, const std::initializer_list<const char*>& default_value) const;
	const Config& loadConfig(const char* key) const;
	Config& loadConfig(const char* key);

	void saveString(const char* key, const char* value);
	void saveString(const char* key, const std::string& value);
	void saveString(const char* key, std::string&& value);
	void saveInt(const char* key, int value);
	void saveFloat(const char* key, float value);
	void saveTuple(const char* key, std::vector<std::string> value);

	void save(std::ostream& file, const char* indentation = "") const;
};


Config loadConfig(const char* filename);
void save(const Config& config, const char* filename);

#endif  // INCLUDED_CONFIG
