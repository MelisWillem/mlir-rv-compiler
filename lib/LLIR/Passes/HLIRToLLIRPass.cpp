#include <memory.h>

#include "HLIR/HLIROps.h"
#include "LLIR/LLIROps.h"
#include "llvm/Support/Debug.h"
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
  return mlir::TypeSwitch<Operation&, LogicalResult> (op)
    .Case<hlir::ConstantOp>([&builder](hlir::ConstantOp op){
      builder.create<ConstantOp>(op.getLoc(), op.getResult().getType(), op.getConstant());
      return LogicalResult::success();
      })
    .Case<hlir::FuncOp>([](hlir::FuncOp funcOp){
        return LogicalResult::success();
    })
    .Case<hlir::Load>([](hlir::Load load){
        return LogicalResult::success();
      })
    .Case<hlir::Store>([](hlir::Store store){
        return LogicalResult::success();
      })
    .Case<hlir::CompareOp>([](hlir::CompareOp cmp){
        return LogicalResult::success();
      })
    .Case<hlir::IfOp>([](hlir::IfOp ifOp){
        return LogicalResult::success();
      })
    .Case<hlir::ReturnOp>([](hlir::ReturnOp returnOp){
        return LogicalResult::success();
      })
    .Case<hlir::AllocaOp>([](hlir::AllocaOp alloca){
        return LogicalResult::success();
      })
    .Case<ModuleOp>([](ModuleOp module){
        // Modules are already builtin, no need to map.
        return LogicalResult::success();
      })
    .Default([](mlir::Operation& op){
      llvm::errs() << "can't translate from hlir->llir op: " << op << "\n";
      return LogicalResult::failure();
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
