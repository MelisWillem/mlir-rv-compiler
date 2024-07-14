#include <cassert>
#include <iostream>
#include <token/tokens.h>
#include <variant>


namespace tokenizer {

std::string Token::strData() const
{
  if(!std::holds_alternative<std::string>(data)){
    std::cout << "not a string type, " << *this <<"\n";
  }
  assert(std::holds_alternative<std::string>(data));
  return std::get<std::string>(data);
}

int Token::intData() const
{
  assert(std::holds_alternative<int>(data));
  return std::get<int>(data);
}

int Token::precedence() const{
  switch (type) {
    case tokenizer::TokenType::DIV:
    case tokenizer::TokenType::MUL:
      return 20;
    case tokenizer::TokenType::MINUS:
    case tokenizer::TokenType::PLUS:
      return 10;
    default: 
      return -1;
  }
}

bool Token::isBinaryOperator() const{
  switch (type) {
    case tokenizer::TokenType::GREATER:
    case tokenizer::TokenType::SMALLER:
    case tokenizer::TokenType::GREATER_EQUAL:
    case tokenizer::TokenType::SMALLER_EQUAL:
    case tokenizer::TokenType::DIV:
    case tokenizer::TokenType::MUL:
    case tokenizer::TokenType::MINUS:
    case tokenizer::TokenType::PLUS:
      return true;
    default:
      return false;
  }
}

bool Token::isCmp() const {
  switch (type) {
    case tokenizer::TokenType::GREATER:
    case tokenizer::TokenType::SMALLER:
    case tokenizer::TokenType::GREATER_EQUAL:
    case tokenizer::TokenType::SMALLER_EQUAL:
      return true;
    default:
      return false;
  }
}

std::string str(const TokenType t){
  switch (t) {
  case TokenType::FUNC:
    return "FUNC";
    break;
  case TokenType::IF:
    return "IF";
    break;
  case TokenType::RETURN:
    return "RETURN";
    break;
  case TokenType::INT:
    return "INT";
    break;
  case TokenType::BOOL:
    return "BOOL";
    break;
  case TokenType::COLON:
    return "COLON";
    break;
  case TokenType::SEMI_COLON:
    return "SEMI_COLON";
    break;
  case TokenType::ARROW:
    return "ARROW";
    break;
  case TokenType::OPEN_PAREN:
    return "OPEN_PAREN";
    break;
  case TokenType::CLOSE_PAREN:
    return "CLOSE_PAREN";
    break;
  case TokenType::OPEN_CURLY:
    return "OPEN_CURLY";
    break;
  case TokenType::CLOSE_CURLY:
    return "CLOSE_CURLY";
    break;
  case TokenType::NUMBER:
    return "NUMBER";
    break;
  case TokenType::IDENTIFIER:
    return "IDENTIFIER";
    break;
  case TokenType::COMMA:
    return "COMMA";
    break;
  case TokenType::ASSIGN:
    return "=";
    break;
  case TokenType::EQUAL:
    return "==";
    break;
  case TokenType::GREATER:
    return "GREATER";
    break;
  case TokenType::GREATER_EQUAL:
    return "GREATER_EQUALS";
    break;
  case TokenType::SMALLER:
    return "SMALLER";
    break;
  case TokenType::SMALLER_EQUAL:
    return "SMALLER_EQUAL";
    break;
  default:
    return "UNKNOWN_TYPE";
    break;
  }
}

std::ostream &operator<<(std::ostream &os, const TokenType t) {
  os << str(t);
  return os;
}

std::ostream &operator<<(std::ostream &os, const Location &loc) {
  os << "{lineNumber=" << loc.lineNumber << ", charNumber=" << loc.charNumber
     << ", offsetBegin=" << loc.offsetBegin << ", offsetEnd=" << loc.offsetEnd
     << "}";
  return os;
}

std::ostream &operator<<(std::ostream &os, const Token &t) {
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