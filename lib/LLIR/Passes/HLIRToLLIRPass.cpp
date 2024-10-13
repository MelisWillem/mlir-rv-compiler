#include "LLIR/LLIRPasses.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include <memory.h>

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "LLIR/HLIRToLLIR.inc.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::llir {

#define GEN_PASS_DEF_HLTOLLIRPASS
#include "LLIR/LLIRPasses.h.inc"

class HLToLLIRPass : public impl::HLToLLIRPassBase<HLToLLIRPass> {
  FrozenRewritePatternSet patterns;

  LogicalResult initialize(MLIRContext *ctx) override {
    // Build the pattern set within the `initialize` to avoid recompiling PDL
    // patterns during each `runOnOperation` invocation.
    RewritePatternSet patternList(ctx);
    populateGeneratedPDLLPatterns(patternList);
    patterns = std::move(patternList);
    return success();
  }

  void runOnOperation() final {
    // Invoke the pattern driver with the provided patterns.
    auto result = applyPatternsAndFoldGreedily(getOperation(), patterns);
    if(result.failed()){
      llvm::errs() << "failed HLIR to LLIR \n";
      signalPassFailure();
    }
  }
};

} // namespace mlir::llir
