#include "RVIR/RVIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <ostream>
#include <sstream>
#include <string>

using namespace mlir;

std::ostream &operator<<(std::ostream &os, rvir::RegisterType reg) {
  if (reg.getId().has_value()) {
    os << "x" << reg.getId().value();
  } else {
    os << "x?";
  }
  return os;
}

std::ostream &operator<<(std::ostream &os,
                         mlir::TypedValue<rvir::RegisterType> reg) {
  os << reg.getType();
  return os;
}

std::string getRVIROpName(mlir::Operation *op) {
  auto fullname = op->getName().getStringRef().str();

  return fullname.substr(5, fullname.size());
}

void ToAsm(std::ostream &os, mlir::Operation &op,
           std::map<mlir::Block *, std::string> &blockNames) {
  if (isa<rvir::ConstantRegOp>(op)) {
    return;
  }
  os << "    ";
  TypeSwitch<Operation &>(op)
      .Case<mlir::rvir::ADDI>([&](mlir::rvir::ADDI op) {
        os << getRVIROpName(op) << " " << op.getResult() << ", " << op.getRd()
           << ", " << op.getImm() << "\n";
      })
      .Case<rvir::SLTI>([&](rvir::SLTI op) {
        os << getRVIROpName(op) << " " << op.getResult() << ", " << op.getRd()
           << ", " << op.getImm() << "\n";
      })
      .Case<rvir::SLT>([&](rvir::SLT op) {
        os << getRVIROpName(op) << " " << op.getResult() << ", " << op.getRs1()
           << ", " << op.getRs2() << "\n";
      })
      .Case<rvir::BEQ>([&](rvir::BEQ op) {
        Block *targetBlock = op->getSuccessor(0);
        os << getRVIROpName(op) << " " << op.getRs1() << ", " << op.getRs2()
           << ", " << blockNames[targetBlock] << "\n";
      })
      .Case<func::ReturnOp>([&](mlir::func::ReturnOp op) { os << "ret\n"; })
      .Default([](mlir::Operation &op) {
        op.emitError() << "Unsupported operation \n";
      });
}

LogicalResult Print(mlir::ModuleOp module, std::ostream &os) {
  for (auto &op : module.getOps()) {
    if (auto funcOp = dyn_cast<mlir::func::FuncOp>(op)) {
      std::map<mlir::Block *, std::string> blockNames;
      for (auto [index, block] : llvm::enumerate(funcOp.getBlocks())) {
        std::string blockName;
        if (block.isEntryBlock()) {
          blockName = funcOp.getName().str();
        } else {
          blockName = funcOp.getName().str() + std::to_string(index);
        }
        blockNames[&block] = blockName;
      }
      for (auto &block : funcOp.getBlocks()) {
        os << blockNames[&block] << ":\n";
        for (auto &op : block) {
          ToAsm(os, op, blockNames);
        }
      }
    } else {
      op.emitError() << "Unsupported operation \n";
      return failure();
    }
  }

  return success();
}