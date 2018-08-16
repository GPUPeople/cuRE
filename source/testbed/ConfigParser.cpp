


#include "ConfigParser.h"


namespace
{
	class DrinkUpMeHearties : public ConfigFile::LexerCallback
	{
	public:
		bool consumeComment(ConfigFile::Stream& stream, ConfigFile::Token t) { return true; }
		bool consumeIdentifier(ConfigFile::Stream& stream, ConfigFile::Token t) { return true; }
		bool consumeIntegerLiteral(ConfigFile::Stream& stream, ConfigFile::Token t) { return true; }
		bool consumeFloatLiteral(ConfigFile::Stream& stream, ConfigFile::Token t) { return true; }
		bool consumeStringLiteral(ConfigFile::Stream& stream, ConfigFile::Token t) { return true; }
		bool consumeOperator(ConfigFile::Stream& stream, ConfigFile::OPERATOR_TOKEN op, ConfigFile::Token t) { return true; }
		bool consumeEOL(ConfigFile::Stream& stream) { return true; }
		void consumeEOF(ConfigFile::Stream& stream) {}
	};


	std::string invalid_token_message(ConfigFile::Stream& stream, ConfigFile::Token t)
	{
		return "invalid token: '" + std::string(t.begin, t.end) + '\'';
	}

	std::string unexpected_token_message(ConfigFile::Stream& stream, ConfigFile::Token t)
	{
		return "unexpected token: '" + std::string(t.begin, t.end) + '\'';
	}

	
	class BasicStreamCallback : public ConfigFile::LexerCallback
	{
	protected:
		BasicStreamCallback() = default;
		BasicStreamCallback(const BasicStreamCallback&) = default;
		BasicStreamCallback& operator =(const BasicStreamCallback&) = default;

		static void invalid_token(ConfigFile::Stream& stream, ConfigFile::Token t)
		{
			auto msg = invalid_token_message(stream, t);
			stream.error(msg);
		}

		static void invalid_token_fatal(ConfigFile::Stream& stream, ConfigFile::Token t)
		{
			auto msg = invalid_token_message(stream, t);
			stream.error(msg);
			throw ConfigFile::parse_error(msg);
		}

		static void unexpected_token(ConfigFile::Stream& stream, ConfigFile::Token t)
		{
			auto msg = unexpected_token_message(stream, t);
			stream.error(msg);
		}

		static void unexpected_token_fatal(ConfigFile::Stream& stream, ConfigFile::Token t)
		{
			auto msg = unexpected_token_message(stream, t);
			stream.error(msg);
			throw ConfigFile::parse_error(msg);
		}

	public:
		bool consumeComment(ConfigFile::Stream& stream, ConfigFile::Token t) { return true; }

		bool consumeIdentifier(ConfigFile::Stream& stream, ConfigFile::Token t)
		{
			invalid_token(stream, t);
			return true;
		}

		bool consumeIntegerLiteral(ConfigFile::Stream& stream, ConfigFile::Token t)
		{
			invalid_token(stream, t);
			return true;
		}

		bool consumeFloatLiteral(ConfigFile::Stream& stream, ConfigFile::Token t)
		{
			invalid_token(stream, t);
			return true;
		}

		bool consumeStringLiteral(ConfigFile::Stream& stream, ConfigFile::Token t)
		{
			invalid_token(stream, t);
			return true;
		}

		bool consumeOperator(ConfigFile::Stream& stream, ConfigFile::OPERATOR_TOKEN op, ConfigFile::Token t)
		{
			invalid_token(stream, t);
			return true;
		}

		bool consumeEOL(ConfigFile::Stream& stream) { return true; }
		void consumeEOF(ConfigFile::Stream& stream) {}
	};


	class Scope : public BasicStreamCallback
	{
	private:
		ConfigFile::ParserCallback& callback;

	public:
		Scope(ConfigFile::ParserCallback& callback)
			: callback(callback)
		{
		}

		bool consumeIdentifier(ConfigFile::Stream& stream, ConfigFile::Token t);

		bool consumeOperator(ConfigFile::Stream& stream, ConfigFile::OPERATOR_TOKEN op, ConfigFile::Token t)
		{
			if (op == ConfigFile::OPERATOR_TOKEN::RBRACE)
			{
				return false;
			}
			stream.error("expected definition");
			return true;
		}

		void consumeEOF(ConfigFile::Stream& stream)
		{
			stream.error("unexpected end of file");
		}
	};


	class Tuple : public BasicStreamCallback
	{
	private:
		ConfigFile::ParserCallback& callback;

		const std::string& id;
		std::vector<std::string> values;
		bool comma;

		void invalidToken(ConfigFile::Stream& stream, ConfigFile::Token t)
		{
			if (!comma)
				stream.error("expected string");
			else
				stream.error("expected ','");
		}

	public:
		Tuple(ConfigFile::ParserCallback& callback, const std::string& id)
			: callback(callback),
			  id(id),
			  comma(true)
		{
		}

		bool consumeStringLiteral(ConfigFile::Stream& stream, ConfigFile::Token t)
		{
			if (comma)
				values.emplace_back(t.begin + 1, t.end - 1);
			else
				stream.error("expected ','");
			comma = false;
			return true;
		}

		bool consumeIntegerLiteral(ConfigFile::Stream& stream, ConfigFile::Token t)
		{
			invalidToken(stream, t);
			comma = false;
			return true;
		}

		bool consumeFloatLiteral(ConfigFile::Stream& stream, ConfigFile::Token t)
		{
			invalidToken(stream, t);
			comma = false;
			return true;
		}

		bool consumeIdentifier(ConfigFile::Stream& stream, ConfigFile::Token t)
		{
			invalidToken(stream, t);
			comma = false;
			return true;
		}

		bool consumeOperator(ConfigFile::Stream& stream, ConfigFile::OPERATOR_TOKEN op, ConfigFile::Token t)
		{
			if (comma)
			{
				stream.error("expected value");
			}
			else
			{
				if (op == ConfigFile::OPERATOR_TOKEN::COMMA)
				{
					comma = true;
					return true;
				}
				else if (op == ConfigFile::OPERATOR_TOKEN::RPARENT)
				{
					callback.addTuple(id, std::move(values));
					return false;
				}
				stream.error("expected ',' or ')'");
			}
			return true;
		}

		void consumeEOF(ConfigFile::Stream& stream)
		{
			stream.error("unexpected end of file in tuple");
		}
	};

	class Value : public BasicStreamCallback
	{
	private:
		ConfigFile::ParserCallback& callback;

		std::string id;
		bool assigned;
		int negate;

		bool expectAssignment(ConfigFile::Stream& stream)
		{
			if (!assigned)
			{
				stream.error("assignment expected");
				return false;
			}
			return true;
		}

	public:
		Value(ConfigFile::ParserCallback& callback, ConfigFile::Token id)
			: callback(callback),
			  id(id.begin, id.end),
			  assigned(false),
			  negate(0)
		{
		}

		bool consumeStringLiteral(ConfigFile::Stream& stream, ConfigFile::Token t)
		{
			if (expectAssignment(stream))
			{
				if (negate)
					stream.error("cannot negate string value");
				callback.addString(id, std::string(t.begin + 1, t.end - 1));
			}
			return false;
		}

		bool consumeIntegerLiteral(ConfigFile::Stream& stream, ConfigFile::Token t)
		{
			if (expectAssignment(stream))
				callback.addInt(id, (negate % 2 == 0 ? 1 : -1) * std::stoi(std::string(t.begin, t.end)));
			return false;
		}

		bool consumeFloatLiteral(ConfigFile::Stream& stream, ConfigFile::Token t)
		{
			if (expectAssignment(stream))
				callback.addFloat(id, (negate % 2 == 0 ? 1.0f : -1.0f) * std::stof(std::string(t.begin, t.end)));
			return false;
		}

		bool consumeIdentifier(ConfigFile::Stream& stream, ConfigFile::Token t)
		{
			stream.error("expected value");
			return true;
		}
		
		bool consumeOperator(ConfigFile::Stream& stream, ConfigFile::OPERATOR_TOKEN op, ConfigFile::Token t)
		{
			if (assigned)
			{
				if (op == ConfigFile::OPERATOR_TOKEN::MINUS)
				{
					++negate;
					return true;
				}
				else if (op == ConfigFile::OPERATOR_TOKEN::LPARENT)
				{
					Tuple value(callback, id);
					consume(stream, value);
					return false;
				}
				else if (op == ConfigFile::OPERATOR_TOKEN::LBRACE)
				{
					Scope scope(callback.addConfig(id));
					consume(stream, scope);
					return false;
				}
				stream.error("expected value");
				return true;
			}
			else
			{
				if (op == ConfigFile::OPERATOR_TOKEN::EQ)
				{
					assigned = true;
					return true;
				}
			}
			stream.error("expected '='");
			return true;
		}

		bool consumeEOL(ConfigFile::Stream& stream)
		{
			return false;
		}
	};


	bool Scope::consumeIdentifier(ConfigFile::Stream& stream, ConfigFile::Token t)
	{
		Value value(callback, t);
		consume(stream, value);
		return true;
	}

	class GlobalScope : public Scope
	{
	public:
		GlobalScope(ConfigFile::ParserCallback& callback)
			: Scope(callback)
		{
		}

		bool consumeOperator(ConfigFile::Stream& stream, ConfigFile::OPERATOR_TOKEN op, ConfigFile::Token t)
		{
			stream.error("expected definition");
			return true;
		}

		void consumeEOF(ConfigFile::Stream& stream)
		{
		}
	};
}

namespace ConfigFile
{
	Stream& parse(Stream& stream, ParserCallback& callback)
	{
		GlobalScope global(callback);
		return consume(stream, global);
	}
}
