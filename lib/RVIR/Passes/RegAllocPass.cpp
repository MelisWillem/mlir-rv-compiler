#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory.h>
#include <memory>
#include <optional>
#include <vector>

#include "RVIR/RVIROps.h"
#include "RVIR/RVIRPasses.h"
#include "RVIR/RVIRTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

mlir::Operation *WithResType(mlir::Operation *op, mlir::Type type, int resIndex,
                             mlir::OpBuilder builder) {
  builder.setInsertionPoint(op);

  return llvm::TypeSwitch<mlir::Operation *, mlir::Operation *>(op)
      .Case<mlir::rvir::ADDI, mlir::rvir::ADD, mlir::rvir::SUB,
            mlir::rvir::SLT>([&](auto specificOp) {
        auto oldResTypes =
            llvm::map_to_vector(op->getResultTypes(), [](auto t) { return t; });
        oldResTypes[resIndex] = type;
        return builder.create<decltype(specificOp)>(
            op->getLoc(), oldResTypes, op->getOperands(), op->getAttrs());
      })
      .Default([](mlir::Operation *op) {
        llvm::errs()
            << "Error: Unsupported operation for register allocation\n";
        llvm::errs() << "Operation: " << *op << "\n";
        return op;
      });
}

namespace RegAllocLinearScan {
using RegType = std::size_t;
struct Interval {
  int begin;
  int end;
  mlir::Value value;
  void dump() const {
    llvm::dbgs() << "[begin=" << begin << " end=" << end << " value=" << value
                 << "]\n";
  }
};

static std::pair<std::vector<Interval>, std::map<void *, RegType>>
getUnassignedIntervals(Block &block, const std::vector<RegType> &resultRegs) {
  std::vector<Interval> intervals;
  std::map<void *, RegType> valueRegAssignment;
  for (auto &arg : block.getArguments()) {
    intervals.push_back({0, 0, arg});
    // the end will be updated on usage in the next for loop
  }
  for (auto [index, op] : llvm::enumerate(block)) {
    if (isa<rvir::ConstantRegOp>(op)) {
      // Constant ops are not used in the register allocation, they by
      // definition already have a register allocated. them.
      continue;
    }
    if (isa<func::ReturnOp>(op)) {
      // return ops have register value already set, so we don't keep track of
      // the interval here.
      for (auto [i, res] : llvm::enumerate(op.getOperands())) {
        // The operands of a returnOp as the return values, these should be
        // preset to the provided registers.
        if (valueRegAssignment.find(res.getAsOpaquePointer()) !=
            valueRegAssignment.end()) {
          valueRegAssignment[res.getAsOpaquePointer()] = resultRegs[i];
        } else {
          assert(i < resultRegs.size() &&
                 "Not enough result registers provided");
          valueRegAssignment.insert({res.getAsOpaquePointer(), resultRegs[i]});
        }
      }
      continue;
    }
    for (auto result : op.getResults()) {
      // If the interval exists, update the end of the interval, otherwise
      // insert a new interval.
      // insert the new interval
      intervals.push_back({(int)index, (int)index, result});
    }
    for (auto operand : op.getOperands()) {
      auto it =
          std::find_if(intervals.begin(), intervals.end(), [&](auto &interval) {
            return interval.value == operand;
          });
      if (it != intervals.end()) {
        // When a function argument is used outside of the initial block,
        // you won't find the value in an interval. Right now we assume that
        // function arguments stay alive in a all parts of the function so we
        // can ignore it.
        it->end = index;
      }
    }
  }
  // the intervals are sorted by the start time, as they are added in that
  // order
  return {intervals, valueRegAssignment};
}

class RegisterAlloc {
  static constexpr RegType numberOfRegisters = 32;
  std::vector<bool> registers;
  RegType numAllocatedRegs = 0;

  [[nodiscard]] bool isFreeRegister(RegType reg) const {
    return reg < numberOfRegisters && !registers[reg];
  }

  void allocRegister(RegType reg) {
    assert(reg >= 0 && reg < numberOfRegisters);
    assert(isFreeRegister(reg));
    registers[reg] = true;
    numAllocatedRegs++;
  }

  std::optional<RegType>
  allocAnyOfRegister(std::initializer_list<RegType> regs) {
    for (auto reg : regs) {
      if (!registers[reg]) {
        allocRegister(reg);
        return reg;
      }
    }
    return {};
  }

public:
  RegisterAlloc() : registers(numberOfRegisters, false) {
    allocRegister(0); // register 0 is reserved -> zero register
    allocRegister(1); // - x1 is the return address
    allocRegister(2); // - x2 is the stack pointer
    allocRegister(3); // - x3 is the global pointer
  }

  [[nodiscard]] int numberOfAvailableRegisters() const {
    return numberOfRegisters - numAllocatedRegs;
  }

  [[nodiscard]] bool hasFreeRegisters() const {
    return numberOfAvailableRegisters() > 0;
  }

  [[nodiscard]] std::optional<RegType> allocateRegister() {
    auto it = std::find(registers.begin(), registers.end(), false);
    if (it != registers.end()) {
      RegType regIndex = std::distance(registers.begin(), it);
      allocRegister(regIndex);
      return regIndex;
    }
    return std::nullopt;
  }

  void freeRegister(RegType reg) {
    if (reg > 0 && reg < numberOfRegisters) {
      registers[reg] = false;
    }
  }

  // RISCV calling convention:
  // - x10 to x17 for the first 8 arguments
  std::optional<RegType> allocReturnArgumentReg() {
    // - x10, x11 are also used for return arguments
    return allocAnyOfRegister({10, 11});
  }

  std::optional<RegType> allocFunctionArgumentReg() {
    // - x10 to x17 for the first 8 arguments
    return allocAnyOfRegister({10, 11, 12, 13, 14, 15, 16, 17});
  }
};

class IntervalTracker {
  // Active intervals are sorted by increasing order of the end time.
  // That way it's easy to clean up the old intervals.
  // We need to track the register used as well here
  std::vector<Interval> activeIntervals;

public:
  // Removes the expired intervals from the active intervals, and returns them
  // from this function.
  std::vector<Interval> expireOldIntervals(int current_begin) {
    // remove all intervals that are not active anymore, i.e. the end time is
    // less than the current begin time.
    auto it = std::upper_bound(activeIntervals.begin(), activeIntervals.end(),
                               current_begin,
                               [](auto &a, Interval &b) { return a < b.end; });
    std::vector<Interval> expiredIntervals;
    std::copy(activeIntervals.begin(), it,
              std::back_inserter(expiredIntervals));

    activeIntervals.erase(activeIntervals.begin(), it);
    return expiredIntervals;
  }

  std::size_t numberOfActiveIntervals() const { return activeIntervals.size(); }

  void addInterval(Interval interval) {
    // insert the interval in the sorted order according to the end time.
    auto it = std::upper_bound(activeIntervals.begin(), activeIntervals.end(),
                               interval,
                               [](auto &a, auto &b) { return a.end < b.end; });
    activeIntervals.insert(it, interval);
  }

  void dump() const {
    llvm::dbgs() << "active intervals: \n";
    for (const auto &interval : activeIntervals) {
      llvm::dbgs() << "   ";
      interval.dump();
    }
  }
};

static std::vector<Interval> LinearScan(const std::vector<Interval> &intervals,
                                        std::map<void *, RegType> &valueMap) {
  std::vector<Interval> assignedRegisters;
  IntervalTracker intervalTracker;
  RegisterAlloc regAlloc;

  // walk over the intervals according to the start time
  for (auto interval : intervals) {
    for (auto expInterval :
         intervalTracker.expireOldIntervals(interval.begin)) {
      if (valueMap.find(expInterval.value.getAsOpaquePointer()) !=
          valueMap.end()) {
        auto reg = valueMap[expInterval.value.getAsOpaquePointer()];
        regAlloc.freeRegister(reg);
      }
    }
    auto regMapIt = valueMap.find(interval.value.getAsOpaquePointer());
    if (regMapIt != valueMap.end()) {
      // value already has a register assigned, most likely this is a value on a
      // return statement.
      continue;
    }
    if (!regAlloc.hasFreeRegisters()) {
      // spill the register, as a heuristic we consider the last active
      // interval to be the best choice for spilling. It has the last end
      // time, so less likely that we need it soon.
      // Implement spilling logic here

    } else {
      auto maybeReg = regAlloc.allocateRegister();
      // Maybe the reg allocator should do the spilling as well,
      // it will prevent this assert.
      assert(maybeReg.has_value() &&
             "the above if statement here should prevent this.");
      assert(regMapIt == valueMap.end() &&
             "the value should not be in the map yet");
      intervalTracker.addInterval(interval);
      assignedRegisters.push_back(interval);
      valueMap.insert({interval.value.getAsOpaquePointer(), maybeReg.value()});
    }
  }

  return assignedRegisters;
}

static LogicalResult runLearScanRegAlloc(Block &block, mlir::OpBuilder &builder,
                                         std::vector<RegType> &inputRegs,
                                         std::vector<RegType> &resultRegs) {
  auto [intervals, valueRegMap] = getUnassignedIntervals(block, resultRegs);
  if (block.isEntryBlock()) {
    // add the input registers to the valueRegMap
    for (auto [index, arg] : llvm::enumerate(block.getArguments())) {
      auto reg = inputRegs[index];
      valueRegMap.insert({arg.getAsOpaquePointer(), reg});
    }
  }
  auto assignedRegisters = LinearScan(intervals, valueRegMap);

  // now apply the register choices to the ir
  for (auto [begin, end, value] : intervals) {
    auto valueRegIt = valueRegMap.find(value.getAsOpaquePointer());
    if (valueRegIt == valueRegMap.end()) {
      llvm::errs() << "unable to allocated register to value: " << value
                   << "\n";
      return failure();
    }
    auto reg = valueRegIt->second;

    if (auto *op = value.getDefiningOp()) {
      for (auto res : op->getResults()) {
        if (res == value) {
          auto resNumber = res.getResultNumber();
          if (auto type = res.getType().dyn_cast<rvir::RegisterType>()) {
            auto newType = rvir::RegisterType::get(builder.getContext(), reg);
            auto newOp = WithResType(op, newType, resNumber, builder);
            op->replaceAllUsesWith(newOp);
            op->erase();
          } else {
            op->emitError() << "Result is not of RVIR type. \n";
          }
        }
      }
    } else {
      // this is a block argument
      assert(llvm::isa<BlockArgument>(value));
      auto arg = cast<BlockArgument>(value);
      const auto argNum = arg.getArgNumber();
      auto newArg = block.insertArgument(
          argNum, mlir::rvir::RegisterType::get(builder.getContext(), reg),
          arg.getLoc());
      arg.replaceAllUsesWith(newArg);
      block.eraseArgument(argNum + 1);
    }
  }

  return mlir::success();
}
}; // namespace RegAllocLinearScan

namespace mlir::rvir {

#define GEN_PASS_DEF_REGALLOC
#include "RVIR/RVIRPasses.h.inc"

class RegAllocPass : public impl::RegAllocBase<RegAllocPass> {
  using impl::RegAllocBase<RegAllocPass>::RegAllocBase;

public:
  void runOnOperation() final {
    auto function = getOperation();
    mlir::DominanceInfo domInfo(function);
    mlir::PostDominanceInfo postDomInfo(function);
    auto &context = getContext();
    auto builder = OpBuilder(&context);

    using namespace RegAllocLinearScan;

    // For now let's just make a list of the registers needed for function
    // arguments and results. And reserve them throught all blocks, we can later
    // on improve the performance by using the dominance/postdominance info to
    // determine what registers should be a alive in what blocks.

    RegisterAlloc regAlloc;
    auto functionType = function.getFunctionType();

    // First reserve the return arguments, as those regsiters can also be used
    // for the function arguments.
    std::vector<RegType> resultRegs;
    std::vector<Type> resultTypes;
    for (std::size_t i = 0; i < functionType.getNumResults(); i++) {
      auto maybeReg = regAlloc.allocReturnArgumentReg();
      assert(maybeReg.has_value());
      resultRegs.push_back(maybeReg.value());
      resultTypes.push_back(RegisterType::get(&context, maybeReg.value()));
    }

    std::vector<RegType> inputRegs;
    std::vector<Type> inputTypes;
    for (std::size_t i = 0; i < functionType.getNumInputs(); i++) {
      auto maybeReg = regAlloc.allocFunctionArgumentReg();
      assert(maybeReg.has_value());
      inputRegs.push_back(maybeReg.value());
      inputTypes.push_back(RegisterType::get(&context, maybeReg.value()));
    }

    for (auto &block : function) {
      if (failed(runLearScanRegAlloc(block, builder, inputRegs, resultRegs))) {
        return signalPassFailure();
      }
    }

    // update the function type
    function.setFunctionType(builder.getFunctionType(inputTypes, resultTypes));
  }
};

} // namespace mlir::rvir