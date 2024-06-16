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

  // symbols
  COLON,
  SEMI_COLON,
  ARROW, // ->
  OPEN_PAREN,
  CLOSE_PAREN,
  OPEN_CURLY,
  CLOSE_CURLY,
  COMMA,

  // operators
  PLUS,
  MINUS,
  MUL,
  DIV,
  GREATER,
  SMALLER,
  GREATER_EQUAL,
  SMALLER_EQUAL,

  // generic
  NUMBER,
  IDENTIFIER
};

struct Token {
  using Data = std::variant<std::monostate, int, std::string>;
  TokenType type;
  Location loc;
  Data data;
};

// Printing of tokens.
std::ostream& operator<<(std::ostream& os, TokenType& t);
std::ostream& operator<<(std::ostream& os, Location& t);
std::ostream& operator<<(std::ostream& os, Token& t);

} // namespace tokenizer