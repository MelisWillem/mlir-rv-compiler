#include "HLIR/HLIRDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "LLIR/LLIRDialect.h"
#include "RVIR/RVIRDialect.h"
#include "RVIR/RVIRPasses.h"
#include "LLIR/LLIRPasses.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::rvir::registerPasses();
  mlir::llir::registerPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::rvir::RVIRDialect, mlir::llir::LLIRDialect, mlir::hlir::HLIRDialect, mlir::arith::ArithDialect, mlir::func::FuncDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "RVIR optimizer driver\n", registry));
}
