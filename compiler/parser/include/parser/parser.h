#pragma once

#include "HLIR/HLIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "token/tokens.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include <filesystem>
#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

namespace parser {

// helper class to keep track of symbols per scope
class SymbolTable {
public:
  class scope {
    std::string name;
    // Symbols contains the address of the value.
    // Due to opaque pointers we need to keep track of the type seperatly.
    std::map<std::string, std::pair<mlir::Value, mlir::Type>> symbols;

  public:
    scope() : name("default"), symbols() {}
    scope(const std::string &name) : name(name), symbols() {}

    std::string Name() const { return name; }

    std::optional<std::pair<mlir::Value, mlir::Type>> Lookup(const std::string &val) const {
      const auto sIt = symbols.find(val);
      if (sIt != symbols.end()) {
        return sIt->second;
      }

      return {}; // can't find it
    }

    void Insert(const std::string &name, mlir::Value val, mlir::Type elementType) {
      symbols.insert({name, {val, elementType}});
    }

    bool Update(const std::string &name, mlir::Value val, mlir::Type elementType) {
      auto symbolIt = symbols.find(name);
      if (symbolIt == symbols.end()) {
        return false; // can't update, symbol is not in the symbol table.
      }
      assert(symbolIt->second.second == elementType && "cannot update a symbol to a new type");
      symbols[name] = {val, elementType};
      return true;
    }
  };

private:
  std::vector<scope> scopes;

public:
  SymbolTable();
  SymbolTable(const SymbolTable &) =
      delete; // do not allow copy, symboltable should only exist once.

  std::optional<std::pair<mlir::Value, mlir::Type>> Lookup(const std::string &name) const;
  void Insert(const std::string &name, mlir::Value val, mlir::Type elementType);
  void PopScope();
  scope CurrentScope() const;
  void InsertScope(const std::string &name);
  bool UpdateSymbol(const std::string &name, mlir::Value val, mlir::Type type);
};

using namespace tokenizer;
class DebugStream;
class Parser {
  const std::vector<Token> &tokens;
  mlir::OpBuilder builder;
  mlir::ModuleOp theModule;
  std::string filename;
  std::string file_text;

  SymbolTable symbolTable;

  int cursor = 0;
  bool debug = true;

  Token Consume();
  Token Peek() const;
  bool PeekIsAnyTokenTypeOf(std::initializer_list<TokenType> token_types) const;
  bool PeekIsType() const;
  bool Expect(TokenType t);
  bool ExpectMany(std::initializer_list<TokenType> tokenTypes);
  bool IsFinal() const;
  void ReportError(const std::string &, std::filesystem::path filepath,
                   int lineNumber);
  DebugStream ReportLog(std::string filename, int lineNumber);

public:
  Parser(const std::vector<Token> &tokens, mlir::MLIRContext *context,
         const std::string &filename, const std::string &file_text, bool debug);

  // create IR
  std::optional<mlir::ModuleOp> Parse();

private:
  mlir::Location Location();
  std::optional<mlir::func::FuncOp> Function();
  std::optional<mlir::func::FuncOp> Prototype(std::vector<mlir::Type>& mlir_args, std::vector<std::string>& name_args);

  const static std::vector<mlir::Type>  empty_mlir_args;
  const static std::vector<std::string> empty_name_args;
  bool Body(mlir::Block*  block, const std::vector<mlir::Type>& mlir_args=empty_mlir_args, const std::vector<std::string>& name_args=empty_name_args);

  std::optional<mlir::Operation *> Statement();
  bool VarDecl();
  std::optional<mlir::cf::CondBranchOp> IfStatement();
  std::optional<mlir::Value> Expression(int precedence = -1);
  mlir::Value CreateBinaryOperator(const Token &OperatorToken, mlir::Value lhs,
                                   mlir::Value rhs);
  mlir::Type Type();
  std::optional<mlir::Value> UnaryExpression();
  void DeclareVar(const std::string &name, mlir::Value val);
  std::optional<mlir::Value> LoadVar(const std::string &name);
};
} // namespace parser
