#include <cstdio>
#include <memory.h>
#include <optional>

#include "RVIR/RVIROps.h"
#include "RVIR/RVIRPasses.h"
#include "RVIR/RVIRTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Debug.h"

namespace mlir::rvir {

#define GEN_PASS_DEF_TORV
#include "RVIR/RVIRPasses.h.inc"

rvir::ConstantRegOp ConstantReg(PatternRewriter &rewriter, Location loc,
                                unsigned id) {
  auto outputType = rewriter.getType<rvir::RegisterType>(id);
  return rewriter.create<rvir::ConstantRegOp>(loc, outputType);
}

rvir::ConstantRegOp NullReg(PatternRewriter& rewriter, Location loc){
  return ConstantReg(rewriter, loc, 0);
}

RegisterType NullRegType(PatternRewriter &rewriter) {
  return rewriter.getType<RegisterType>(0);
}

RegisterType VirtRegister(PatternRewriter &rewriter) {
  return rewriter.getType<RegisterType>(std::nullopt);
}

ADDI Move(PatternRewriter &rewriter, Location loc,
          TypedValue<RegisterType> from, RegisterType to) {
  return rewriter.create<ADDI>(loc,
                               to,    // rd: output register
                               0,     // imm == 0 with a move
                               from); // rd: read/input register
}

class CmpIPattern : public OpRewritePattern<arith::CmpIOp> {
public:
  CmpIPattern(PatternBenefit benefit, MLIRContext *context)
      : OpRewritePattern(context, 1) {}

  LogicalResult
  matchAndRewrite(arith::CmpIOp cmpOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto lhsRegVal = cmpOp.getLhs().getDefiningOp<rvir::RegValue>();
    auto rhsRegVal = cmpOp.getRhs().getDefiningOp<rvir::RegValue>();
    if (!lhsRegVal || !rhsRegVal) {
      return failure();
    }
    auto lhsReg = lhsRegVal.getInput();
    auto rhsReg = rhsRegVal.getInput();

    // allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    // enum class CmpIPredicate : uint64_t {
    //   eq = 0,
    //   ne = 1,
    //   slt = 2,
    //   sle = 3,
    //   sgt = 4,
    //   sge = 5,
    //   ult = 6,
    //   ule = 7,
    //   ugt = 8,
    //   uge = 9,
    // };
    auto outputType = cmpOp.getResult().getType();
    // kind of a hack, always inject a slt instruction even if the predicate is
    // not sle.

    auto slt = rewriter.create<rvir::SLT>(
        cmpOp->getLoc(),
        rewriter.getType<rvir::RegisterType>(std::nullopt), // ::mlir::Type rd,
        lhsReg,                                             // ::mlir::Value rs1
        rhsReg                                              // ::mlir::Value rs2
    );

    auto castedSlt =
        rewriter.create<rvir::RegValue>(cmpOp->getLoc(), outputType, slt);
    cmpOp->replaceAllUsesWith(castedSlt);
    cmpOp.erase();
    return success();
  }
};

class CondBranchOpPattern : public OpRewritePattern<cf::CondBranchOp> {
public:
  CondBranchOpPattern(PatternBenefit benefit, MLIRContext *context)
      : OpRewritePattern(context, 1) {}

  LogicalResult matchAndRewrite(cf::CondBranchOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    auto cond = op.getCondition();
    auto regValCond = cond.getDefiningOp<RegValue>();

    if (!regValCond) {
      return failure();
    }

    auto *oldBlock = rewriter.getBlock();

    auto condValue = regValCond.getInput();
    rewriter.create<rvir::BEQ>(op->getLoc(), condValue, NullReg(rewriter, op->getLoc()),
                               op.getFalseDest());

    // if the true dest was the next block, maybe we can skip this?
    auto *newBlock = oldBlock->splitBlock(op);

    // otherwise always jump to the true condition
    // J or JAL with rd=0 is a terminal instruction for a basicblock.
    // builder.create<rvir::JAL>
    // uint32_t rs1, uint32_t rd, ::mlir::Block *succ);
    rewriter.setInsertionPointToStart(newBlock);
    rewriter.create<rvir::JALR>(op->getLoc(), op.getCondition(),
                                NullReg(rewriter, op.getLoc()), op.getTrueDest());

    op.erase();

    return success();
  }
};

class RegValValRegRVIRPattern : public OpRewritePattern<rvir::ValueReg> {
public:
  RegValValRegRVIRPattern(PatternBenefit benefit, MLIRContext *context)
      : OpRewritePattern(context, 1) {}

  /// This overload constructs a pattern that only matches operations with the
  /// root name of `MyOp`.
  void initialize() { setDebugName("RegValValRegRVIR pattern"); }

  LogicalResult matchAndRewrite(rvir::ValueReg valReg,
                                PatternRewriter &rewriter) const override {
    // do the transformation:
    // from: y = (ValReg(RegVal(x)) : RegisterType
    // to: y = x : RegisterType
    auto regVal = valReg.getInput().getDefiningOp<rvir::RegValue>();
    if (!regVal) {
      return rewriter.notifyMatchFailure(valReg, [](Diagnostic &diag) {
        diag << "input of op is not regVal";
      });
    }

    auto xValue = regVal.getInput();
    auto yValue = valReg.getResult();
    if (yValue.getType().getId() != xValue.getType().getId()) {
      // we can get rid of the conversion, but we need an extra move as the
      // registers don't line up
      rewriter.setInsertionPointAfterValue(yValue);
      xValue = Move(rewriter, valReg->getLoc(), xValue, yValue.getType());
    }

    // - valReg has result type RegisterType
    // - regVal.getInput() has type RegisterType
    rewriter.replaceAllUsesWith(yValue, xValue);

    if (valReg->use_empty()) {
      rewriter.eraseOp(valReg);
    }
    if (regVal.use_empty()) {
      rewriter.eraseOp(regVal);
    }
    return success();
  }
};

class ConstantRVIRPattern : public RewritePattern {
public:
  ConstantRVIRPattern(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(arith::ConstantOp::getOperationName(), benefit,
                       context) {}

  void initialize() { setDebugName("ConstantRVIR pattern"); }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // The `matchAndRewrite` method performs both the matching and the mutation.
    // Note that the match must reach a successful point before IR mutation may
    // take place.
    auto constantOp = dyn_cast<arith::ConstantOp>(op);
    if (!constantOp) {
      return rewriter.notifyMatchFailure(op, "Not a arith::ConstantOp.");
    }

    auto typedAttr = constantOp.getValue();
    if (auto intAttr = dyn_cast<IntegerAttr>(typedAttr)) {
      auto APIntVal = intAttr.getValue();
      auto val = APIntVal.getZExtValue();
      rewriter.setInsertionPointAfterValue(constantOp);
      // pseudo move instruction
      auto newAdd =
          rewriter.create<ADDI>(constantOp->getLoc(),
                                VirtRegister(rewriter), // rd: output register
                                val,                    // imm
                                nullptr // rd: set to null register
          );
      auto newRegVal = rewriter.create<rvir::RegValue>(
          newAdd->getLoc(), constantOp.getResult().getType(), newAdd);
      constantOp.replaceAllUsesWith(newRegVal.getResult());
      constantOp.erase();
      return success();
    }
    return rewriter.notifyMatchFailure(
        constantOp, "Only support integer constants right now \n");
  }
};

class ToRVPass : public impl::ToRVBase<ToRVPass> {
  FrozenRewritePatternSet patterns;

  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet rewritePatterns(context);
    rewritePatterns // add<TPattern>(benefit, context)
        .add<CmpIPattern>(1, context)
        .add<CondBranchOpPattern>(1, context)
        .add<ConstantRVIRPattern>(1, context)
        .add<RegValValRegRVIRPattern>(1, context);

    patterns = FrozenRewritePatternSet(std::move(rewritePatterns), std::nullopt,
                                       std::nullopt);

    return success();
  }

  void runOnOperation() final {
    auto function = getOperation();
    OpBuilder builder(&getContext());

    // remove the arguments from the function.
    // TODO: find out how to change the callers.
    builder.setInsertionPointToStart(
        &*function.getFunctionBody().getBlocks().begin());
    // builder.create<rvir::RegValue>(builder.getUnknownLoc(), 2); // x2 is the
    // stack pointer

    // RISCV calling convention:
    // - x10 to x17 for the first 8 arguments
    // - x10, x11 are also used for return arguments
    // - x1 is the return address
    // - x2 is the stack pointer
    // - x3 is the global pointer

    auto regIndex =
        10; // as the registers for fnction arguments are straightforward they
            // are set to the correct value straight away.
    auto numResults = function.getFunctionType().getNumResults();
    assert(numResults <= 2 && "more then 2 results not supported yet");
    auto numResultsInRegs = std::min<decltype(numResults)>(numResults, 2);
    std::vector<Type> resTypes;
    // for(auto resType : function.getFunctionType().getResults()){
    for (std::size_t i = 0; i < numResultsInRegs; i++) {
      resTypes.push_back(builder.getType<rvir::RegisterType>(regIndex));
      regIndex++;
    }

    auto numInputs = function.getFunctionType().getNumInputs();
    auto numArgumentsInRegs =
        std::min<decltype(numInputs)>(8 - resTypes.size(), numInputs);
    assert(numInputs <= numArgumentsInRegs &&
           "more then 8 arguments/results not supported yet");

    std::vector<rvir::RegisterType> inputTypes;
    for (std::size_t i = 0; i < numArgumentsInRegs; i++) {
      // arguments are passed in registers x10 to x17 (aka a0 to a7)
      inputTypes.push_back(builder.getType<rvir::RegisterType>(regIndex));
      regIndex++;
    }
    function.setFunctionType(builder.getFunctionType(
        llvm::map_to_vector(inputTypes, [](auto t) -> Type { return t; }),
        resTypes));
    // TODO: deal the inputs that did not fit in the registers, so when
    // numInputs - numArgumentsInRegs != 0

    // fix the inputs
    auto &initBlock = function.getBody().getBlocks().front();
    for (std::size_t i = 0; i < initBlock.getNumArguments(); ++i) {
      auto arg = initBlock.getArgument(i);
      auto newArg = initBlock.insertArgument(i, inputTypes[i], arg.getLoc());
      auto convertedArg =
          builder.create<RegValue>(newArg.getLoc(), arg.getType(), newArg);
      arg.replaceAllUsesWith(convertedArg);
      initBlock.eraseArgument(i +
                              1); // the old argument is now after the new one
    }

    // fix the outputs
    function->walk([&builder](func::ReturnOp returnOp) {
      auto regIndex = 10;
      assert(returnOp->getOperands().size() <= 2 &&
             "No more then 2 results supported at tihs time.");
      for (auto [index, val] :
           llvm::enumerate(returnOp->getOperands().take_front(2))) {
        builder.setInsertionPointAfterValue(val);
        auto convertVal = builder.create<ValueReg>(
            val.getLoc(), builder.getType<RegisterType>(regIndex), val);
        returnOp.setOperand(index, convertVal);
        regIndex++;
      }
    });

    GreedyRewriteConfig config;
    config.maxIterations = 10;
    config.enableRegionSimplification = true;
    if (failed(applyPatternsAndFoldGreedily(function, patterns, config))) {
      function->emitError() << "Fialed to apply patterns in ToRVPass \n";
      return signalPassFailure();
    }
  }
};

} // namespace mlir::rvir
