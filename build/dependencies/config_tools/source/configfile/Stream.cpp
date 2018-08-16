


#include <string>
#include <stdexcept>

#include <configfile/Stream.h>


namespace
{
	void invalidCharacter(configfile::Stream& stream)
	{
		std::string msg = std::string("invalid input character: '") + *stream.current() + '\'';
		stream.error(msg.c_str());
		throw configfile::lexer_error(msg);
	}


	bool isalpha(char c)
	{
		return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '$' || c == '_' || c == '.' || c == '@';
	}

	bool isdigit_dec(char c)
	{
		return c >= '0' && c <= '9';
	}

	bool isdigit_oct(char c)
	{
		return c >= '0' && c <= '7';
	}

	bool isdigit_hex(char c)
	{
		return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
	}

	bool isdigit_bin(char c)
	{
		return c == '0' || c == '1';
	}

	bool isalnum(char c)
	{
		return isalpha(c) || isdigit_dec(c);
	}

	configfile::Stream& next(configfile::Stream& stream)
	{
		stream.get();
		if (stream.eof())
		{
			const char* msg = "unexpected end of file";
			stream.error(msg);
			throw configfile::lexer_error(msg);
		}
		return stream;
	}

	template <typename F>
	configfile::Stream& readSequence(configfile::Stream& stream, F f)
	{
		while (!stream.eof() && f(*stream.current()))
			stream.get();
		return stream;
	}

	configfile::Stream& readLineComment(configfile::Stream& stream)
	{
		return readSequence(stream, [](char c) { return c != '\n'; });
	}

	configfile::Stream& readBlockComment(configfile::Stream& stream)
	{
		char c0 = *next(stream).current();
		while (true)
		{
			char c1 = *next(stream).current();
			if (c0 == '*' && c1 == '/')
			{
				break;
			}
			c0 = c1;
		}
		return stream;
	}

	configfile::Stream& readIdentifier(configfile::Stream& stream)
	{
		return readSequence(stream, isalnum);
	}

	configfile::Stream& readStringLiteral(configfile::Stream& stream)
	{
		bool escape = false;
		while (true)
		{
			if (*stream.current() == '"' && !escape)
			{
				stream.get();
				break;
			}
			else if (*stream.current() == '\n')
			{
				stream.error("line break in string literal");
				throw configfile::lexer_error("line break in string literal");
			}
			else if (stream.eof())
			{
				stream.error("end of file in string literal");
				throw configfile::lexer_error("end of file in string literal");
			}

			if (*stream.current() == '\\' && !escape)
			{
				next(stream);
				escape = true;
				continue;
			}

			next(stream);
			escape = false;
		}
		return stream;
	}

	bool readFloatLiteralExponent(configfile::Stream& stream, const char* begin, configfile::LexerCallback& callback)
	{
		if (!stream.eof() && (*stream.current() == 'e' || *stream.current() == 'E'))
		{
			next(stream);
			if (*stream.current() == '+' || *stream.current() == '-')
				next(stream);
			if (!isdigit_dec(*stream.current()))
				invalidCharacter(stream);
			readSequence(stream, isdigit_dec);
		}

		return callback.consumeFloatLiteral(stream, configfile::Token(begin, stream.current()));
	}

	bool readFloatLiteralFraction(configfile::Stream& stream, const char* begin, configfile::LexerCallback& callback)
	{
		return readFloatLiteralExponent(readSequence(stream, isdigit_dec), begin, callback);
	}

	bool readNumberLiteral(configfile::Stream& stream, const char* begin, configfile::LexerCallback& callback)
	{
		readSequence(stream, isdigit_dec);

		if (!stream.eof())
		{
			if (*stream.current() == '.')
			{
				stream.get();
				return readFloatLiteralFraction(stream, begin, callback);
			}
		}

		return callback.consumeIntegerLiteral(stream, configfile::Token(begin, stream.current()));
	}

	bool readNumberLiteralPrefix(configfile::Stream& stream, const char* begin, configfile::LexerCallback& callback)
	{
		if (*stream.current() == 'x' || *stream.current() == 'X')
		{
			const char* begin_prefix = stream.get();
			if (readSequence(stream, isdigit_hex).current() - begin_prefix <= 1)
				invalidCharacter(stream);
			return callback.consumeIntegerLiteral(stream, configfile::Token(begin, stream.current()));
		}
		return readNumberLiteral(stream, begin, callback);
	}

	configfile::OPERATOR_TOKEN getTwoCharacterOp(configfile::Stream& stream, configfile::OPERATOR_TOKEN op1, char c2, configfile::OPERATOR_TOKEN op2)
	{
		if (!stream.eof() && (*stream.current() == c2))
		{
			stream.get();
			return op2;
		}
		return op1;
	}

	configfile::OPERATOR_TOKEN getTwoCharacterOp(configfile::Stream& stream, configfile::OPERATOR_TOKEN op1, char c2, configfile::OPERATOR_TOKEN op2, char c3, configfile::OPERATOR_TOKEN op3)
	{
		if (!stream.eof())
		{
			if (*stream.current() == c2)
			{
				stream.get();
				return op2;
			}
			if (*stream.current() == c3)
			{
				stream.get();
				return op3;
			}
		}
		return op1;
	}

	configfile::OPERATOR_TOKEN getOperator(configfile::Stream& stream)
	{
		switch(*stream.get())
		{
			case '+':
				return configfile::OPERATOR_TOKEN::PLUS;
			case '-':
				return getTwoCharacterOp(stream, configfile::OPERATOR_TOKEN::MINUS, '>', configfile::OPERATOR_TOKEN::ARROW);
			case '*':
				return configfile::OPERATOR_TOKEN::ASTERISK;
			case '^':
				return configfile::OPERATOR_TOKEN::CIRCUMFLEX;
			case '~':
				return configfile::OPERATOR_TOKEN::TILDE;
			case '(':
				return configfile::OPERATOR_TOKEN::LPARENT;
			case ')':
				return configfile::OPERATOR_TOKEN::RPARENT;
			case '{':
				return configfile::OPERATOR_TOKEN::LBRACE;
			case '}':
				return configfile::OPERATOR_TOKEN::RBRACE;
			case '[':
				return configfile::OPERATOR_TOKEN::LBRACKET;
			case ']':
				return configfile::OPERATOR_TOKEN::RBRACKET;
			case '?':
				return configfile::OPERATOR_TOKEN::QUEST;
			case ':':
				return configfile::OPERATOR_TOKEN::COLON;
			case ',':
				return configfile::OPERATOR_TOKEN::COMMA;
			case ';':
				return configfile::OPERATOR_TOKEN::SEMICOLON;
			case '<':
				return getTwoCharacterOp(stream, configfile::OPERATOR_TOKEN::LT, '=', configfile::OPERATOR_TOKEN::LEQ, '<', configfile::OPERATOR_TOKEN::LL);
			case '>':
				return getTwoCharacterOp(stream, configfile::OPERATOR_TOKEN::GT, '=', configfile::OPERATOR_TOKEN::GEQ, '<', configfile::OPERATOR_TOKEN::GG);
			case '&':
				return getTwoCharacterOp(stream, configfile::OPERATOR_TOKEN::AND, '&', configfile::OPERATOR_TOKEN::AAND);
			case '|':
				return getTwoCharacterOp(stream, configfile::OPERATOR_TOKEN::OR, '&', configfile::OPERATOR_TOKEN::OOR);
			case '=':
				return getTwoCharacterOp(stream, configfile::OPERATOR_TOKEN::EQ, '=', configfile::OPERATOR_TOKEN::EEQ);
			case '!':
				return getTwoCharacterOp(stream, configfile::OPERATOR_TOKEN::BANG, '=', configfile::OPERATOR_TOKEN::NEQ);
		}

		invalidCharacter(stream);
		return configfile::OPERATOR_TOKEN::INVALID;
	}
}

namespace configfile
{
	Stream::Stream(const char* begin, const char* end, const char* name, Log& log)
		: ptr(begin),
		  end(end),
		  stream_name(name),
		  log(log)
	{
		lines.push_back(begin);
	}

	Stream& consume(LexerCallback& callback, Stream& stream)
	{
		while (!stream.eof())
		{
			switch (*stream.current())
			{
				case '/':
				{
					const char* begin = stream.get();

					if (!stream.eof())
					{
						if (*stream.current() == '/')
						{
							readLineComment(stream);
							if (!callback.consumeComment(stream, Token(begin, stream.current())))
								return stream;
							break;
						}
						else if (*stream.current() == '*')
						{
							readBlockComment(stream);
							if (!callback.consumeComment(stream, Token(begin, stream.current())))
								return stream;
							break;
						}
					}

					if (!callback.consumeOperator(stream, OPERATOR_TOKEN::SLASH, Token(begin, stream.current())))
						return stream;
					break;
				}

				case '.':
				{
					const char* begin = stream.get();

					if (!stream.eof())
					{
						if (isdigit_dec(*stream.current()))
						{
							if (!readFloatLiteralFraction(stream, begin, callback))
								return stream;
							break;
						}
					}

					if (!callback.consumeOperator(stream, OPERATOR_TOKEN::DOT, Token(begin, stream.current())))
						return stream;
					break;
				}

				case '%':
				{
					const char* begin = stream.get();

					if (!stream.eof() && isalnum(*stream.current()))
					{
						readIdentifier(stream);
						if (!callback.consumeIdentifier(stream, Token(begin, stream.current())))
							return stream;
						break;
					}

					if (!callback.consumeOperator(stream, OPERATOR_TOKEN::PERCENT, Token(begin, stream.current())))
						return stream;
					break;
				}

				case '0':
				{
					const char* begin = stream.get();

					if (!readNumberLiteralPrefix(stream, begin, callback))
						return stream;
					break;
				}

				case '"':
				{
					const char* begin = stream.get();
					readStringLiteral(stream);
					if (!callback.consumeStringLiteral(stream, Token(begin, stream.current())))
							return stream;
					break;
				}

				case '\n':
					stream.get();
					if (!callback.consumeEOL(stream))
						return stream;
					break;
				case '\r':
				case '\t':
				case ' ':
					stream.get();
					break;

				default:
				{
					const char* begin = stream.current();

					if (isdigit_dec(*stream.current()))
					{
						if (!readNumberLiteral(stream, begin, callback))
							return stream;
						break;
					}
					else if (isalnum(*stream.current()))
					{
						readIdentifier(stream);
						if (!callback.consumeIdentifier(stream, Token(begin, stream.current())))
							return stream;
						break;
					}

					configfile::OPERATOR_TOKEN op = getOperator(stream);

					if (!callback.consumeOperator(stream, op, Token(begin, stream.current())))
						return stream;
					break;
				}
			}
		}
		callback.consumeEOL(stream);
		callback.consumeEOF(stream);
		return stream;
	}

	size_t Stream::getCurrentLineNumber()
	{
		return lines.size();
	}

	ptrdiff_t Stream::getCurrentColumn()
	{
		return ptr - lines.back();
	}

	void Stream::warning(const char* message)
	{
		log.warning(message, stream_name, getCurrentLineNumber(), getCurrentColumn());
	}

	void Stream::warning(const std::string& message)
	{
		log.warning(message, stream_name, getCurrentLineNumber(), getCurrentColumn());
	}

	void Stream::error(const char* message)
	{
		log.error(message, stream_name, getCurrentLineNumber(), getCurrentColumn());
	}

	void Stream::error(const std::string& message)
	{
		log.error(message, stream_name, getCurrentLineNumber(), getCurrentColumn());
	}

	//Token token(OPERATOR_TOKEN op)
	//{
	//	switch (op)
	//	{
	//		case OPERATOR_TOKEN::PLUS:
	//			return Token("+", 1);
	//		case OPERATOR_TOKEN::MINUS:
	//			return Token("-", 1);
	//		case OPERATOR_TOKEN::ASTERISK:
	//			return Token("*", 1);
	//		case OPERATOR_TOKEN::SLASH:
	//			return Token("/", 1);
	//		case OPERATOR_TOKEN::CIRCUMFLEX:
	//			return Token("^", 1);
	//		case OPERATOR_TOKEN::TILDE:
	//			return Token("~", 1);
	//		case OPERATOR_TOKEN::LPARENT:
	//			return Token("(", 1);
	//		case OPERATOR_TOKEN::RPARENT:
	//			return Token(")", 1);
	//		case OPERATOR_TOKEN::LBRACKET:
	//			return Token("[", 1);
	//		case OPERATOR_TOKEN::RBRACKET:
	//			return Token("]", 1);
	//		case OPERATOR_TOKEN::LBRACE:
	//			return Token("{", 1);
	//		case OPERATOR_TOKEN::RBRACE:
	//			return Token("}", 1);
	//		case OPERATOR_TOKEN::QUEST:
	//			return Token("?", 1);
	//		case OPERATOR_TOKEN::DOT:
	//			return Token(".", 1);
	//		case OPERATOR_TOKEN::COLON:
	//			return Token(":", 1);
	//		case OPERATOR_TOKEN::COMMA:
	//			return Token(",", 1);
	//		case OPERATOR_TOKEN::SEMICOLON:
	//			return Token(";", 1);
	//		case OPERATOR_TOKEN::EQ:
	//			return Token("=", 1);
	//		case OPERATOR_TOKEN::EEQ:
	//			return Token("==", 2);
	//		case OPERATOR_TOKEN::NEQ:
	//			return Token("!=", 2);
	//		case OPERATOR_TOKEN::LT:
	//			return Token("<", 1);
	//		case OPERATOR_TOKEN::LEQ:
	//			return Token("<=", 2);
	//		case OPERATOR_TOKEN::LL:
	//			return Token("<<", 2);
	//		case OPERATOR_TOKEN::GT:
	//			return Token(">", 1);
	//		case OPERATOR_TOKEN::GEQ:
	//			return Token(">=", 2);
	//		case OPERATOR_TOKEN::GG:
	//			return Token(">>", 2);
	//		case OPERATOR_TOKEN::AND:
	//			return Token("&", 1);
	//		case OPERATOR_TOKEN::AAND:
	//			return Token("&&", 2);
	//		case OPERATOR_TOKEN::OR:
	//			return Token("|", 1);
	//		case OPERATOR_TOKEN::OOR:
	//			return Token("||", 2);
	//		case OPERATOR_TOKEN::BANG:
	//			return Token("!", 1);
	//		case OPERATOR_TOKEN::PERCENT:
	//			return Token("%", 1);
	//		case OPERATOR_TOKEN::ARROW:
	//			return Token("->", 1);
	//	}
	//	throw std::runtime_error("invalid enum");
	//}
}
