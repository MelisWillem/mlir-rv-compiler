#include <iostream>
#include <token/tokens.h>
#include <variant>


namespace tokenizer {

std::ostream &operator<<(std::ostream &os, TokenType &t) {
  switch (t) {
  case TokenType::FUNC:
    os << "FUNC";
    break;
  case TokenType::IF:
    os << "IF";
    break;
  case TokenType::RETURN:
    os << "RETURN";
    break;
  case TokenType::INT:
    os << "INT";
    break;
  case TokenType::BOOL:
    os << "BOOL";
    break;
  case TokenType::COLON:
    os << "COLON";
    break;
  case TokenType::SEMI_COLON:
    os << "SEMI_COLON";
    break;
  case TokenType::ARROW:
    os << "ARROW";
    break;
  case TokenType::OPEN_PAREN:
    os << "OPEN_PAREN";
    break;
  case TokenType::CLOSE_PAREN:
    os << "CLOSE_PAREN";
    break;
  case TokenType::OPEN_CURLY:
    os << "OPEN_CURLY";
    break;
  case TokenType::CLOSE_CURLY:
    os << "CLOSE_CURLY";
    break;
  case TokenType::NUMBER:
    os << "NUMBER";
    break;
  case TokenType::IDENTIFIER:
    os << "IDENTIFIER";
    break;
  case TokenType::COMMA:
    os << "COMMA";
    break;
  case TokenType::GREATER:
    os << "GREATER";
    break;
  case TokenType::GREATER_EQUAL:
    os << "GREATER_EQUALS";
    break;
  case TokenType::SMALLER:
    os << "SMALLER";
    break;
  case TokenType::SMALLER_EQUAL:
    os << "SMALLER_EQUAL";
    break;
  default:
    os << "UNKNOWN_TYPE";
    break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, Location &loc) {
  os << "{lineNumber=" << loc.lineNumber << ", charNumber=" << loc.charNumber
     << ", offsetBegin=" << loc.offsetBegin << ", offsetEnd=" << loc.offsetEnd
     << "}";
  return os;
}

std::ostream &operator<<(std::ostream &os, Token &t) {
  os << "{type=" << t.type << ", loc=" << t.loc;
  if(std::holds_alternative<int>(t.data)){
    os << ",int_data=" << std::get<int>(t.data);
  }
  if(std::holds_alternative<std::string>(t.data)){
    os << ",string_data=" << std::get<std::string>(t.data);
  }
  os << "}";
  return os;
}

} // namespace tokenizer