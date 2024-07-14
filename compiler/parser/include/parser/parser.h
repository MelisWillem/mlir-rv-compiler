#pragma once

#include "HLIR/HLIROps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "token/tokens.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include <filesystem>
#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

namespace mlir::hlir {
class FuncOp;
}

namespace parser {

// helper class to keep track of symbols per scope
class SymbolTable {
public:
  class scope {
    std::string name;
    std::map<std::string, mlir::Value> symbols;

  public:
    scope() : name("default"), symbols() {}
    scope(const std::string &name) : name(name), symbols() {}

    std::string Name() const { return name; }

    std::optional<mlir::Value> Lookup(const std::string &val) const {
      const auto sIt = symbols.find(val);
      if (sIt != symbols.end()) {
        return sIt->second;
      }

      return {}; // can't find it
    }

    void Insert(const std::string &name, mlir::Value val) {
      symbols.insert({name, val});
    }

    bool Update(const std::string &name, mlir::Value val) {
      if (symbols.find(name) == symbols.end()) {
        return false; // can't update, symbol is not in the symbol table.
      }
      symbols[name] = val;
      return true;
    }
  };

private:
  std::vector<scope> scopes;

public:
  SymbolTable();
  SymbolTable(const SymbolTable &) =
      delete; // do not allow copy, symboltable should only exist once.

  std::optional<mlir::Value> Lookup(const std::string &name) const;
  void Insert(const std::string &name, mlir::Value val);
  void PopScope();
  scope CurrentScope() const;
  void InsertScope(const std::string &name);
  bool UpdateSymbol(const std::string &name, mlir::Value val);
};

using namespace tokenizer;
class DebugStream;
class Parser {
  const std::vector<Token> &tokens;
  mlir::MLIRContext *context;
  mlir::OpBuilder builder;
  mlir::ModuleOp theModule;
  std::string filename;
  std::string file_text;

  SymbolTable symbolTable;

  int cursor;
  bool debug_enabled = true;

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
         const std::string &filename, const std::string &file_text);

  // create IR
  std::optional<mlir::ModuleOp> Parse();

private:
  mlir::Location Location();
  std::optional<mlir::hlir::FuncOp> Function();
  std::optional<mlir::hlir::FuncOp> Prototype();
  bool Body(mlir::Block &region);

  std::optional<mlir::Operation *> Statement();
  bool VarDecl();
  std::optional<mlir::hlir::IfOp> IfStatement();
  std::optional<mlir::Value> Expression(int precedence = -1);
  mlir::Value CreateBinaryOperator(const Token &OperatorToken, mlir::Value lhs,
                                   mlir::Value rhs);
  mlir::Type Type();
  std::optional<mlir::Value> UnaryExpression();
  void DeclareVar(const std::string &name, mlir::Value val);
};
} // namespace parser