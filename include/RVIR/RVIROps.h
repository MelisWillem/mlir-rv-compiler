#ifndef VRIR_VRIROPS_H
#define VRIR_VRIROPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "RVIR/RVIRTypes.h"

#define GET_OP_CLASSES
#include "RVIR/RVIROps.h.inc"

#endif // VRIR_VRIROPS_H
