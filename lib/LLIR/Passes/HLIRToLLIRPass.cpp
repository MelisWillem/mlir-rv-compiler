#include <memory.h>

#include "HLIR/HLIROps.h"
#include "LLIR/LLIROps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "HLIR/HLIRPasses.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::llir {

#define GEN_PASS_DEF_HLTOLLIRPASS
#include "LLIR/LLIRPasses.h.inc"

static LogicalResult MapOperation(Operation& op, mlir::OpBuilder& builder){
  builder.setInsertionPoint(&op);
  return mlir::TypeSwitch<Operation&, LogicalResult> (op)
    .Default([](mlir::Operation& op){
      return LogicalResult::success();
      });
}

class HLToLLIRPass : public impl::HLToLLIRPassBase<HLToLLIRPass> {
  LogicalResult initialize(MLIRContext *ctx) override {
    return success();
  }

  void runOnOperation() final {
    auto module = getOperation();
    mlir::OpBuilder builder(&getContext());
    auto result = LogicalResult::success();
    module.walk([&builder, &result](Operation* operation){
      if(operation==nullptr){
        llvm::errs() << "detected nullptr operation... \n" ;
        result = LogicalResult::failure();
        return mlir::WalkResult::interrupt();
      }
      if(MapOperation(*operation, builder).failed()){
        result = LogicalResult::failure();
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
      });
    if(result.failed()){
      return signalPassFailure();
    }
  }
};

} // namespace mlir::llir
