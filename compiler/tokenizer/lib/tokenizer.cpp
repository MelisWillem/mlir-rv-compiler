#include "token/tokens.h"
#include <cctype>
#include <map>
#include <sstream>
#include <stdexcept>
#include <token/tokenizer.h>

namespace tokenizer {

std::optional<char> Tokenizer::Peek() const {
  if (!IsFinal()) {
    return sourceCode[cursor];
  }
  return {};
}

char Tokenizer::Consume() {
  auto output = sourceCode[cursor];
  cursor = cursor + 1;
  charNumber = charNumber + 1;
  return output;
}

bool Tokenizer::IsFinal() const { return sourceCode.length() <= cursor; }

void Tokenizer::Identifier() {
  auto maybeChar = Peek();
  while (maybeChar.has_value() && std::isalnum(*maybeChar)) {
    Consume();
    maybeChar = Peek();
  }

  auto strIdentifier = sourceCode.substr(cursorBegin, cursor - cursorBegin);
  static std::map<std::string, TokenType> knownIdentifiers{
      {"func", TokenType::FUNC},     {"if", TokenType::IF},
      {"return", TokenType::RETURN}, {"int", TokenType::INT},
      {"bool", TokenType::BOOL},
  };

  auto knownId = knownIdentifiers.find(strIdentifier);
  if (knownId != knownIdentifiers.end()) {
    PushbackToken(knownId->second);
    return;
  }

  PushbackToken(TokenType::IDENTIFIER, strIdentifier);
}

void Tokenizer::Numerical() {
  auto maybeChar = Peek();
  while (maybeChar.has_value() && std::isdigit(*maybeChar)) {
    Consume();
    maybeChar = Peek();
  }

  auto subStr = sourceCode.substr(cursorBegin, cursor - cursorBegin);
  int number = 0;
  try {
    number = std::stoi(subStr);
  } catch (const std::invalid_argument &) {
    std::stringstream ss;
    ss << "Failed to parse number from string='" << subStr << "'";
    this->RegisterError(ss.str());
  }
  PushbackToken(TokenType::NUMBER, number);
}

bool Tokenizer::Parse() {
  cursorBegin = 0;
  cursor = 0;
  lineNumber = 1;
  charNumber = 1;
  while (!IsFinal() && !error.has_value()) {
    auto c = Consume();

    switch (c) {
    case ' ':
      break;
    case '\n':
      lineNumber = lineNumber + 1;
      charNumber = 1;
      break;
    case '{':
      PushbackToken(TokenType::OPEN_CURLY);
      break;
    case '}':
      PushbackToken(TokenType::CLOSE_CURLY);
      break;
    case '(':
      PushbackToken(TokenType::OPEN_PAREN);
      break;
    case ')':
      PushbackToken(TokenType::CLOSE_PAREN);
      break;
    case ';':
      PushbackToken(TokenType::SEMI_COLON);
      break;
    case ':':
      PushbackToken(TokenType::COLON);
      break;
    case ',':
      PushbackToken(TokenType::COMMA);
      break;
    case '-': {
      auto maybeNext = Peek();
      if (maybeNext == '>') {
        Consume();
        PushbackToken(TokenType::ARROW);
      } else if (maybeNext.has_value() && std::isdigit(*maybeNext)) {
        // this is negative number
        Numerical();
      } else {
        PushbackToken(TokenType::MINUS);
      }
    } break;
    case '+':
      PushbackToken(TokenType::PLUS);
      break;
    case '*':
      PushbackToken(TokenType::MUL);
      break;
    case '/':
      if (Peek() == '/') {
        // comment line, ignore till the end of the line
        while (Peek() != '\n') {
          Consume();
        }
      } else {
        PushbackToken(TokenType::DIV);
      }
      break;
    case '>':
      if (Peek() == '=') {
        PushbackToken(TokenType::GREATER_EQUAL);
      } else {
        PushbackToken(TokenType::GREATER);
      }
      break;
    case '<':
      if (Peek() == '=') {
        PushbackToken(TokenType::SMALLER_EQUAL);
      } else {
        PushbackToken(TokenType::SMALLER);
      }
      break;
    default:
      if (std::isalpha(c)) {
        Identifier();
      } else if (std::isdigit(c)) {
        Numerical();
      } else {
        std::stringstream ss;
        ss << "unknown character '" << c << "'";

        RegisterError(ss.str());
      }
    }

    cursorBegin = cursor;
    charNumberBegin = charNumber;
  }

  return !error.has_value();
}

} // namespace tokenizer