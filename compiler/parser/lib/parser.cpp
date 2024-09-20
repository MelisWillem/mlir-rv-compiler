#include "parser/parser.h"
#include "HLIR/HLIRAttributes.h"
#include "HLIR/HLIRDialect.h"
#include "HLIR/HLIREnums.h"
#include "HLIR/HLIROps.h"
#include "HLIR/HLIRTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "token/tokens.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <ostream>
#include <sstream>
#include <sys/cdefs.h>
#include <vector>

using namespace tokenizer;

namespace parser {

void SymbolTable::InsertScope(const std::string &name) {
  scopes.push_back({name});
}

bool SymbolTable::UpdateSymbol(const std::string &name, mlir::Value val,
                               mlir::Type type) {
  for (auto &scope : scopes) {
    if (scope.Update(name, val, type)) {
      return true;
    }
  }
  return false;
}

struct ScopeGuard {
  SymbolTable &table;
  std::string scopename;

  ScopeGuard(SymbolTable &table, std::string scopename)
      : table(table), scopename(scopename) {
    table.InsertScope(scopename);
  }

  ~ScopeGuard() {
    assert(table.CurrentScope().Name() == scopename);
    table.PopScope();
  }
};

SymbolTable::scope SymbolTable::CurrentScope() const {
  assert(!scopes.empty());
  return scopes.back();
}

std::optional<std::pair<mlir::Value, mlir::Type>>
SymbolTable::Lookup(const std::string &name) const {
  for (auto &scope : llvm::reverse(scopes)) {
    auto maybeSymbol = scope.Lookup(name);
    if (maybeSymbol.has_value()) {
      return *maybeSymbol;
    }
  }

  return {};
}

void SymbolTable::Insert(const std::string &name, mlir::Value val,
                         mlir::Type elementType) {
  assert(!scopes.empty());
  scopes.back().Insert(name, val, elementType);
}

SymbolTable::SymbolTable() { scopes.push_back({}); }

void SymbolTable::PopScope() {
  assert(!scopes.empty());
  scopes.pop_back();
}

class DebugBuf : public std::stringbuf {
  std::filesystem::path file;
  int linenumber;
  bool enable;

public:
  DebugBuf(std::filesystem::path file, int linenumber, bool enable)
      : file(file), linenumber(linenumber), enable(enable) {}
  ~DebugBuf() { sync(); }
  virtual int sync() override {
    if (!enable) {
      return 0;
    }
    if (str().empty()) {
      return 0;
    }
    std::cout << "DEBUG: " << file.filename().string() << ":" << linenumber
              << " " << this->str() << "\n";
    str("");
    return 0;
  }
};

class DebugStream : public std::ostream {
  DebugBuf buf;

public:
  DebugStream(bool enable, std::filesystem::path file, int linenumber)
      : std::ostream(&buf), buf(file, linenumber, enable) {}
};

const std::vector<mlir::Type> Parser::empty_mlir_args = {};
const std::vector<std::string> Parser::empty_name_args = {};

mlir::Location Parser::Location() { return builder.getUnknownLoc(); }

Parser::Parser(const std::vector<Token> &tokens, mlir::MLIRContext *context,
               const std::string &filename, const std::string &file_text,
               bool debug)
    : tokens(tokens), builder(context),
      theModule(mlir::ModuleOp::create(builder.getUnknownLoc())),
      filename(filename), file_text(file_text), debug(debug) {
  context->loadDialect<mlir::hlir::HLIRDialect>();
}

#define Error(msg) ReportError(msg, __FILE__, __LINE__)
#define Log() ReportLog(__FILE__, __LINE__)

void Parser::ReportError(const std::string &msg, std::filesystem::path filepath,
                         int lineNumber) {
  // print text around the token.
  auto begin = tokens[cursor].loc.offsetBegin;
  auto end = tokens[cursor].loc.offsetEnd;

  auto text = file_text.substr(begin, end + 1);

  std::cout << "ERROR: " << filepath.filename().string() << ":" << lineNumber
            << " " << msg << "\n";
  std::cout << "on " << text << "with tokentype=" << tokens[cursor].type
            << "\n";
}

DebugStream Parser::ReportLog(std::string filename, int lineNumber) {
  return DebugStream(debug, filename, lineNumber);
}

bool Parser::Expect(TokenType t) {
  if (IsFinal()) {
    Error("Unexpected ending");
  }
  auto next_token = Peek();
  auto out = next_token.type == t;
  if (!out) {
    std::stringstream ss;
    ss << "Expected token of type=" << t << " but got " << next_token
       << "instead.";
    Error(ss.str());
    return false;
  }

  return out;
}

bool Parser::ExpectMany(std::initializer_list<TokenType> tokens) {
  for (auto t : tokens) {
    if (!Expect(t)) {
      return false;
    }
  }
  return true;
}

Token Parser::Consume() {
  auto output = tokens[cursor];
  cursor = cursor + 1;
  return output;
}

Token Parser::Peek() const { return tokens[cursor]; }

bool Parser::PeekIsAnyTokenTypeOf(
    std::initializer_list<TokenType> token_types) const {
  if (IsFinal()) {
    return false;
  }
  return std::find(token_types.begin(), token_types.end(), Peek().type) !=
         token_types.end();
}

bool Parser::PeekIsType() const {
  return PeekIsAnyTokenTypeOf({TokenType::BOOL, TokenType::INT});
}

bool Parser::IsFinal() const {
  return cursor >= static_cast<int>(tokens.size());
}

std::optional<mlir::ModuleOp> Parser::Parse() {
  Log() << "Parse ModuleOp with " << tokens.size() << " token.\n";

  builder.setInsertionPointToEnd(theModule.getBody());

  // consume 1 token -> switch...
  while (!IsFinal()) {
    auto t = Consume();
    switch (t.type) {
    case TokenType::FUNC:
      Function();
      break;
    default:
      Error("Unexpected Token ");
      return theModule;
    }
  }
  Log() << "done parsing \n";
  return theModule;
}

mlir::Type Parser::Type() {
  switch (Peek().type) {
  case TokenType::INT:
    Consume();
    return builder.getI32Type();
  case TokenType::BOOL:
    Consume();
    return builder.getI1Type();
  default:
    Error("Epected typed");
    return {};
  }
}

std::optional<mlir::hlir::FuncOp> Parser::Function() {
  Log() << "Parse Function";
  ScopeGuard sg(symbolTable, "function");

  std::vector<mlir::Type> mlir_args;
  std::vector<std::string> name_args;
  auto func = Prototype(mlir_args, name_args);
  if (!func.has_value()) {
    return {};
  }
  auto &function_block = func->getRegion();
  if (!Body(function_block, mlir_args, name_args)) {
    return {};
  }

  Log() << "Finished parsing Function";
  return func;
}

std::optional<mlir::hlir::FuncOp>
Parser::Prototype(std::vector<mlir::Type> &mlir_args,
                  std::vector<std::string> &name_args) {
  Log() << "Parsing prototype ";
  // example: "func food(bar:int)"
  // is the func already consumed by the outer loop?
  if (Peek().type != TokenType::IDENTIFIER) {
    return {};
  }
  auto name = std::get<std::string>(Consume().data);
  Log() << "Function name=" << name;
  if (Peek().type != TokenType::OPEN_PAREN) {
    return {};
  }
  Consume();

  while (Peek().type == TokenType::IDENTIFIER) {
    auto argName = Consume().strData();
    name_args.push_back(argName);

    if (Peek().type != TokenType::COLON) {
      Error("Epected colon after argument name.");
      return {};
    }
    Consume();

    mlir_args.push_back(Type());

    if (Peek().type == TokenType::COMMA) {
      Consume(); // consume comma
    }
  }

  Log() << "Parsed " << mlir_args.size() << " args";

  if (Peek().type != TokenType::CLOSE_PAREN) {
    Error("Expected close brace to end function type.");
    return {};
  }
  Consume();

  std::optional<mlir::Type> mlir_return_type;
  if (Peek().type == TokenType::ARROW) {
    Consume();
    mlir_return_type = Type();
    Log() << "Parsed return type";
  }

  auto funcType = builder.getFunctionType(
      mlir_args, mlir_return_type.has_value() ? *mlir_return_type : nullptr);
  auto arg_attrs = builder.getArrayAttr({});
  auto return_attrs = builder.getArrayAttr({});
  auto funcOp = builder.create<mlir::hlir::FuncOp>(
      builder.getUnknownLoc(), name, funcType, arg_attrs, return_attrs);
  Log() << "created FuncOp from prototype";

  return funcOp;
}

bool Parser::Body(mlir::Region &region,
                  const std::vector<mlir::Type> &mlir_args,
                  const std::vector<std::string> &name_args) {
  Log() << "Parsing block";
  auto *old_insert_block = builder.getBlock();
  auto block = builder.createBlock(&region);

  for (auto [name, type] : llvm::zip(name_args, mlir_args)) {
    auto arg = block->addArgument(type, Location());
    Log() << "add arg:";
    DeclareVar(name, arg);
  }
  Log() << "Added arguments to block and symbol table.";

  builder.setInsertionPointToEnd(block);
  Log() << "block before=" << old_insert_block << "\n";
  Log() << "block while=" << block << "\n";
  if (Peek().type != TokenType::OPEN_CURLY) {
    Error("Expected open curly braces to start block.");
    return false;
  }
  Consume(); // consume open_curly

  while (Peek().type != TokenType::CLOSE_CURLY) {
    auto stmt = Statement();
    if (!stmt) {
      Error("Unable to parse statement in block.");
      return false;
    }
  }

  if (Peek().type == TokenType::CLOSE_CURLY) {
    Consume();
  } else {
    Error("Blocks should end with }.");
  }

  builder.setInsertionPointToEnd(old_insert_block);

  Log() << "block after=" << builder.getBlock() << "\n";
  return true;
}

std::optional<mlir::hlir::IfOp> Parser::IfStatement() {
  Log() << "Parsing if statement";
  if (Peek().type != TokenType::IF) {
    Error("Expected if statement.");
    return {};
  }
  Consume();
  auto cond = Expression();
  if (!cond.has_value()) {
    Error("invalid condition in if statement");
    return {};
  }
  Log() << "Create If statement without block";
  auto IfStmt = builder.create<mlir::hlir::IfOp>(Location(), *cond);

  ScopeGuard sg{symbolTable, "ifexpr"};
  Body(IfStmt.getRegion());
  return {IfStmt};
}

bool Parser::VarDecl() {
  Log() << "Parsing variable declartion";
  if (Peek().type != TokenType::VAR) {
    Error("Var declartion should start with the var keyword");
    return false;
  }
  Consume();

  if (Peek().type != TokenType::IDENTIFIER) {
    Error("variable named expected after var.");
    return false;
  }
  auto identifier_name = Consume().strData();

  if (Peek().type != TokenType::COLON) {
    Error("Colon expected after identifier");
    return false;
  }
  Consume();

  if (Peek().type != TokenType::IDENTIFIER) {
    Error("Typename expected.");
    return false;
  }
  Consume();

  if (Peek().type != TokenType::ASSIGN) {
    Error("Assign expected after variable declration.");
    return false;
  }
  Consume();

  auto expr = Expression(); // intializer
  if (!expr.has_value()) {
    Error("Unable to evaluate init expression of variable declaration");
    return false;
  }

  if (Peek().type != TokenType::SEMI_COLON) {
    Error("statement should end with semicolon.");
    return false;
  }
  Consume();

  // if the symbol is already defined we have a problem
  if (symbolTable.CurrentScope().Lookup(identifier_name).has_value()) {
    Log() << "duplicate symbol: " << identifier_name;
    Error("Symbol already exists.");
    return false;
  }

  DeclareVar(identifier_name, *expr);
  return true;
}

std::optional<mlir::Operation *> Parser::Statement() {
  Log() << "Parsing Statement";
  if (Peek().type == TokenType::VAR) {
    auto varDeclSuccess = VarDecl();
    if (!varDeclSuccess) {
      return {};
    }
  }
  if (Peek().type == TokenType::IF) {
    auto ifstmt = IfStatement();
    if (ifstmt.has_value()) {
      return {ifstmt.value()};
    } else {
      return {};
    }
  }
  if (Peek().type == TokenType::RETURN) {
    Consume(); // return keyword
    if (Peek().type == TokenType::SEMI_COLON) {
      Consume(); // semicolon
      return builder.create<mlir::hlir::ReturnOp>(Location(), nullptr);
    }
    auto returnExpr = Expression();
    if (!returnExpr.has_value()) {
      Error("Expected expression or semi colom after return.");
      return {};
    }

    if (Peek().type != TokenType::SEMI_COLON) {
    }
    Consume(); // semicolon
    return builder.create<mlir::hlir::ReturnOp>(Location(), *returnExpr);
  }
  Log() << "Parsing lhs statement";
  if (Peek().type != TokenType::IDENTIFIER) {
    Error("Expected identifier in lhs of statement.");
    return {};
  }
  auto identifierToken = Consume();

  switch (Peek().type) {
  case TokenType::ASSIGN: {
    Consume();
    auto rhs = Expression();
    if (!Expect(TokenType::SEMI_COLON)) {
      Error("Assign should end with semi colon");
      return {};
    }
    // if the symbol doesnt exist we cant set it.
    auto maybeSymbol = symbolTable.Lookup(identifierToken.strData());
    if (!maybeSymbol.has_value()) {
      Log() << "redefinition of symbol:" << identifierToken.strData();
      Error("Already defined symbol");
      return {};
    }
    auto [_, type] = *maybeSymbol;
    symbolTable.UpdateSymbol(identifierToken.strData(), *rhs, type);
    break;
  }
  case TokenType::SEMI_COLON: {
    Consume();
    return {};
  }
  case TokenType::OPEN_CURLY: {
    // when using an expression inside if condition
    return {};
  }
  default: {
    Error("Expected either asign or semicolon after expression.");
    return {};
  }
  }
  return {};
}

mlir::Value Parser::CreateBinaryOperator(const Token &OperatorToken,
                                         mlir::Value lhs, mlir::Value rhs) {
  using namespace mlir::hlir;
  if (OperatorToken.isCmp()) {
    // CmpTypeAttr
    CmpType type;
    switch (OperatorToken.type) {
    case TokenType::GREATER:
      type = CmpType::greather;
      break;
    case TokenType::SMALLER:
      type = CmpType::smaller;
      break;
    case TokenType::GREATER_EQUAL:
      type = CmpType::eqgreather;
      break;
    case TokenType::SMALLER_EQUAL:
      type = CmpType::eqsmaller;
      break;
    case TokenType::EQUAL:
      type = CmpType::equal;
      break;
    default:
      Error("Unsupported compare operation.");
      return nullptr;
    }
    return builder.create<mlir::hlir::CompareOp>(
        Location(), builder.getI1Type(),
        builder.getAttr<mlir::hlir::CmpTypeAttr>(type), lhs, rhs);
  }
  Error("Unsupported binary operation.");
  return nullptr;
}

std::optional<mlir::Value> Parser::UnaryExpression() {
  auto loc = Location();
  auto token = Consume();
  switch (token.type) {
  case TokenType::IDENTIFIER: {
    return LoadVar(token.strData());
  } break;
  case TokenType::NUMBER: {
    return builder.create<mlir::hlir::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(token.intData()));
  } break;
  case TokenType::BOOL: {
    assert(token.intData() > -1 && token.intData() < 2 &&
           "Bool is just 1 bit, the int data should be 0 or 1");
    return builder.create<mlir::hlir::ConstantOp>(
        loc, builder.getI1Type(), builder.getI32IntegerAttr(token.intData()));
  } break;
  default:
    Error("Expected an unary expression.");
    return {};
  }

  return {};
}

std::optional<mlir::Value> Parser::Expression(int precidence) {
  Log() << "Parsing expression";
  // expression start with number or identifier
  mlir::Value lhs = nullptr;

  auto maybeLhs = UnaryExpression();
  if (!maybeLhs.has_value()) {
    Error("Expected number or identifier at start of expression");
    return {};
  }
  Log() << "lhs is ok";
  lhs = *maybeLhs;

  while (Peek().isBinaryOperator()) {
    auto op_token = Consume();
    auto op_precidence = op_token.precedence();
    if (op_precidence > precidence) {
      Log() << "op_precidence > precidence";
      auto rhs = Expression(op_precidence - 1);
      if (!rhs.has_value()) {
        return {};
      }
      lhs = CreateBinaryOperator(op_token, lhs, *rhs);
    } else {
      Log() << "op_precidence <= precidence";
      auto rhs = UnaryExpression();
      if (!rhs.has_value()) {
        return {};
      }
      lhs = CreateBinaryOperator(op_token, lhs, *rhs);
    }
  }

  return {lhs};
}

void Parser::DeclareVar(const std::string &name, mlir::Value val) {
  Log() << "Declare variable name=" << name << "\n";
  auto type = val.getType();
  auto ptr_type = mlir::hlir::PointerType::get(builder.getContext());
  auto addr =
      builder.create<mlir::hlir::AllocaOp>(Location(), ptr_type, type, name)
          .getResult();
  builder.create<mlir::hlir::Store>(Location(), val, addr);

  symbolTable.Insert(name, addr, type);
}

std::optional<mlir::Value> Parser::LoadVar(const std::string &name) {
  Log() << "Loading variable: " << name << "\n";
  auto maybe_addr = symbolTable.Lookup(name);
  if (!maybe_addr.has_value()) {
    Error(std::string("cant find variable") + name);
    return {};
  }

  // todo: fix type here
  return builder.create<mlir::hlir::Load>(Location(), maybe_addr->first.getType(),
                                          maybe_addr->first);
}

} // namespace parser