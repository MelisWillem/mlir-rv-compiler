#pragma once
#include<iosfwd>
#include<string>
#include<variant>

namespace tokenizer {
struct Location {
  unsigned int lineNumber;
  unsigned int charNumber;

  // index in the input string
  unsigned long offsetBegin; // first char
  unsigned long offsetEnd;   // char after
};
enum class TokenType {
  // keywords
  FUNC,
  IF,
  RETURN,
  INT,
  BOOL,
  VAR,

  // symbols
  COLON,
  SEMI_COLON,
  ARROW, // ->
  OPEN_PAREN,
  CLOSE_PAREN,
  OPEN_CURLY,
  CLOSE_CURLY,
  COMMA,
  ASSIGN,

  // operators
  PLUS,
  MINUS,
  MUL,
  DIV,
  GREATER,
  SMALLER,
  GREATER_EQUAL,
  SMALLER_EQUAL,
  EQUAL, // ==

  // generic
  NUMBER,
  IDENTIFIER
};

struct Token {
  using Data = std::variant<std::monostate, int, std::string>;
  TokenType type;
  Location loc;
  Data data;

  std::string strData() const;  
  int intData() const;  
  int precedence() const;
  bool isBinaryOperator() const;
  bool isCmp() const;
};

std::string str(const TokenType t);

// Printing of tokens.
std::ostream& operator<<(std::ostream& os, const TokenType t);
std::ostream& operator<<(std::ostream& os, const Location& t);
std::ostream& operator<<(std::ostream& os, const Token& t);

} // namespace tokenizer