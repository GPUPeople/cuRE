


#include <configfile/Parser.h>


namespace
{
	class DrinkUpMeHearties : public configfile::LexerCallback
	{
	public:
		bool consumeComment(configfile::Stream& stream, configfile::Token t) { return true; }
		bool consumeIdentifier(configfile::Stream& stream, configfile::Token t) { return true; }
		bool consumeIntegerLiteral(configfile::Stream& stream, configfile::Token t) { return true; }
		bool consumeFloatLiteral(configfile::Stream& stream, configfile::Token t) { return true; }
		bool consumeStringLiteral(configfile::Stream& stream, configfile::Token t) { return true; }
		bool consumeOperator(configfile::Stream& stream, configfile::OPERATOR_TOKEN op, configfile::Token t) { return true; }
		bool consumeEOL(configfile::Stream& stream) { return true; }
		void consumeEOF(configfile::Stream& stream) {}
	};


	std::string invalid_token_message(configfile::Stream& stream, configfile::Token t)
	{
		return "invalid token: '" + std::string(t.begin, t.end) + '\'';
	}

	std::string unexpected_token_message(configfile::Stream& stream, configfile::Token t)
	{
		return "unexpected token: '" + std::string(t.begin, t.end) + '\'';
	}


	class BasicStreamCallback : public configfile::LexerCallback
	{
	protected:
		BasicStreamCallback() = default;
		BasicStreamCallback(const BasicStreamCallback&) = default;
		BasicStreamCallback& operator =(const BasicStreamCallback&) = default;

		static void invalid_token(configfile::Stream& stream, configfile::Token t)
		{
			auto msg = invalid_token_message(stream, t);
			stream.error(msg);
		}

		static void invalid_token_fatal(configfile::Stream& stream, configfile::Token t)
		{
			auto msg = invalid_token_message(stream, t);
			stream.error(msg);
			throw configfile::parse_error(msg);
		}

		static void unexpected_token(configfile::Stream& stream, configfile::Token t)
		{
			auto msg = unexpected_token_message(stream, t);
			stream.error(msg);
		}

		static void unexpected_token_fatal(configfile::Stream& stream, configfile::Token t)
		{
			auto msg = unexpected_token_message(stream, t);
			stream.error(msg);
			throw configfile::parse_error(msg);
		}

	public:
		bool consumeComment(configfile::Stream& stream, configfile::Token t) { return true; }

		bool consumeIdentifier(configfile::Stream& stream, configfile::Token t)
		{
			invalid_token(stream, t);
			return true;
		}

		bool consumeIntegerLiteral(configfile::Stream& stream, configfile::Token t)
		{
			invalid_token(stream, t);
			return true;
		}

		bool consumeFloatLiteral(configfile::Stream& stream, configfile::Token t)
		{
			invalid_token(stream, t);
			return true;
		}

		bool consumeStringLiteral(configfile::Stream& stream, configfile::Token t)
		{
			invalid_token(stream, t);
			return true;
		}

		bool consumeOperator(configfile::Stream& stream, configfile::OPERATOR_TOKEN op, configfile::Token t)
		{
			invalid_token(stream, t);
			return true;
		}

		bool consumeEOL(configfile::Stream& stream) { return true; }
		void consumeEOF(configfile::Stream& stream) {}
	};


	class Scope : public BasicStreamCallback
	{
		configfile::ParserCallback* callback;

	public:
		Scope(configfile::ParserCallback* callback)
			: callback(callback)
		{
		}

		~Scope()
		{
			if (callback)
				callback->leaveNode();
		}

		bool consumeIdentifier(configfile::Stream& stream, configfile::Token t);

		bool consumeOperator(configfile::Stream& stream, configfile::OPERATOR_TOKEN op, configfile::Token t)
		{
			if (op == configfile::OPERATOR_TOKEN::RBRACE)
			{
				return false;
			}
			stream.error("expected definition");
			return true;
		}

		void consumeEOF(configfile::Stream& stream)
		{
			stream.error("unexpected end of file");
		}
	};


	class Tuple : public BasicStreamCallback
	{
		configfile::ParserCallback* callback;

		const std::string& id;
		std::vector<std::string> values;
		bool comma;

		void invalidToken(configfile::Stream& stream, configfile::Token t)
		{
			if (!comma)
				stream.error("expected string");
			else
				stream.error("expected ','");
		}

	public:
		Tuple(configfile::ParserCallback* callback, const std::string& id)
			: callback(callback),
			  id(id),
			  comma(true)
		{
		}

		bool consumeStringLiteral(configfile::Stream& stream, configfile::Token t)
		{
			if (comma)
				values.emplace_back(t.begin + 1, t.end - 1);
			else
				stream.error("expected ','");
			comma = false;
			return true;
		}

		bool consumeIntegerLiteral(configfile::Stream& stream, configfile::Token t)
		{
			invalidToken(stream, t);
			comma = false;
			return true;
		}

		bool consumeFloatLiteral(configfile::Stream& stream, configfile::Token t)
		{
			invalidToken(stream, t);
			comma = false;
			return true;
		}

		bool consumeIdentifier(configfile::Stream& stream, configfile::Token t)
		{
			invalidToken(stream, t);
			comma = false;
			return true;
		}

		bool consumeOperator(configfile::Stream& stream, configfile::OPERATOR_TOKEN op, configfile::Token t)
		{
			if (comma)
			{
				stream.error("expected value");
			}
			else
			{
				if (op == configfile::OPERATOR_TOKEN::COMMA)
				{
					comma = true;
					return true;
				}
				else if (op == configfile::OPERATOR_TOKEN::RPARENT)
				{
					if (callback)
						callback->addTuple(id, std::move(values));
					return false;
				}
				stream.error("expected ',' or ')'");
			}
			return true;
		}

		void consumeEOF(configfile::Stream& stream)
		{
			stream.error("unexpected end of file in tuple");
		}
	};

	class Value : public BasicStreamCallback
	{
		configfile::ParserCallback* callback;

		std::string id;
		bool assigned;
		int negate;

		bool expectAssignment(configfile::Stream& stream)
		{
			if (!assigned)
			{
				stream.error("assignment expected");
				return false;
			}
			return true;
		}

	public:
		Value(configfile::ParserCallback* callback, configfile::Token id)
			: callback(callback),
			  id(id.begin, id.end),
			  assigned(false),
			  negate(0)
		{
		}

		bool consumeStringLiteral(configfile::Stream& stream, configfile::Token t)
		{
			if (expectAssignment(stream))
			{
				if (negate)
					stream.error("cannot negate string value");
				if (callback)
					callback->addString(id, std::string(t.begin + 1, t.end - 1));
			}
			return false;
		}

		bool consumeIntegerLiteral(configfile::Stream& stream, configfile::Token t)
		{
			if (expectAssignment(stream) && callback)
				callback->addInt(id, (negate % 2 == 0 ? 1 : -1) * std::stoi(std::string(t.begin, t.end)));
			return false;
		}

		bool consumeFloatLiteral(configfile::Stream& stream, configfile::Token t)
		{
			if (expectAssignment(stream) && callback)
				callback->addFloat(id, (negate % 2 == 0 ? 1.0f : -1.0f) * std::stof(std::string(t.begin, t.end)));
			return false;
		}

		bool consumeIdentifier(configfile::Stream& stream, configfile::Token t)
		{
			stream.error("expected value");
			return true;
		}
		
		bool consumeOperator(configfile::Stream& stream, configfile::OPERATOR_TOKEN op, configfile::Token t)
		{
			if (assigned)
			{
				switch (op)
				{
				case configfile::OPERATOR_TOKEN::MINUS:
					++negate;
					return true;

				case configfile::OPERATOR_TOKEN::PLUS:
					return true;

				case configfile::OPERATOR_TOKEN::LPARENT:
					{
						Tuple value(callback, id);
						consume(value, stream);
						return false;
					}

				case configfile::OPERATOR_TOKEN::LBRACE:
					{
						auto cb = callback ? callback->enterNode(id) : nullptr;
						Scope scope(cb);
						consume(scope, stream);
						return false;
					}
				}
				stream.error("expected value");
				return true;
			}
			else
			{
				if (op == configfile::OPERATOR_TOKEN::EQ)
				{
					assigned = true;
					return true;
				}
			}
			stream.error("expected '='");
			return true;
		}

		bool consumeEOL(configfile::Stream& stream)
		{
			return false;
		}
	};


	bool Scope::consumeIdentifier(configfile::Stream& stream, configfile::Token t)
	{
		Value value(callback, t);
		consume(value, stream);
		return true;
	}

	class GlobalScope : public Scope
	{
	public:
		GlobalScope(configfile::ParserCallback* callback)
			: Scope(callback)
		{
		}

		bool consumeOperator(configfile::Stream& stream, configfile::OPERATOR_TOKEN op, configfile::Token t)
		{
			stream.error("expected definition");
			return true;
		}

		void consumeEOF(configfile::Stream& stream)
		{
		}
	};
}

namespace configfile
{
	Stream& parse(ParserCallback& callback, Stream& stream)
	{
		GlobalScope global(&callback);
		return consume(global, stream);
	}
}
