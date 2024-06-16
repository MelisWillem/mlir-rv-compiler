#include <optional>
#include <string>
#include <token/tokens.h>
#include <variant>
#include <vector>

namespace tokenizer {

class Tokenizer {
public:
  struct Error {
    std::string message;
    unsigned long cursor;
    unsigned long cursorBegin;
  };

private:
  std::vector<Token> tokens;
  std::optional<Error> error;
  std::string sourceCode;

  void PushbackToken(TokenType type, Token::Data data = std::monostate()) {
    Location loc{lineNumber, charNumberBegin, cursorBegin, cursor};
    tokens.push_back({type, loc, data});
  }

  // cursor is closed interval,
  // both cursor and cursor begin belong
  // to the interval.
  unsigned long cursor = 0;
  unsigned long cursorBegin = 0;
  unsigned int charNumber = 0;
  unsigned int charNumberBegin = 0;
  unsigned int lineNumber = 0;

  void RegisterError(std::string message) {
    error = {message, cursor, cursorBegin};
  }

  void ResetCursor() { cursorBegin = cursor + 1; }

public:
  Tokenizer(std::string sourceCode)
      : tokens(), error(), sourceCode(sourceCode) {}

  // returns true if success
  // return false is fail -> getError() retuns the error;
  // All the parsed tokens are pushed into an internal vector
  // the can be accessed using the getTokens() method.
  bool Parse();

  std::optional<char> Peek() const;

  char Consume();

  bool IsFinal() const;

  void Identifier();

  void Numerical();

  const std::vector<Token> &getTokens() { return tokens; }

  std::optional<Error> getError() { return error; }
};

} // namespace tokenizer