#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "LLIR/LLIRPasses.h"

namespace mlir::rvir {
#define GEN_PASS_DEF_LLIRSWITCHBARFOO
#include "LLIR/LLIRPasses.h.inc"

namespace {

} // namespace
} // namespace mlir::LLIR
