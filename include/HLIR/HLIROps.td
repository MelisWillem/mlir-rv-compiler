#ifndef LIR_OPS
#define LIR_OPS

include "HLIR/HLIRTypes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "HLIREnums.td"
include "HLIRAttributes.td"

//===----------------------------------------------------------------------===//
// Base HLIR operation definition.
//===----------------------------------------------------------------------===//

class HLIR_Op<string mnemonic, list<Trait> traits = []> :
        Op<HLIR_Dialect, mnemonic, traits>;

def CallOp : HLIR_Op<"Call">{
  let arguments = (ins FlatSymbolRefAttr:$sym_name);
}

def ReturnOp : HLIR_Op<"Return", [Terminator]>{ // maybe also add ReturnLike from ControlFlowInterfaces.td
  let arguments = (outs Optional<AnyType>:$operand);
}

def ForOp : HLIR_Op<"For">{
  let regions = (region AnyRegion:$body);
}

def YieldOp : HLIR_Op<"Yield", [Terminator]>{ // maybe also add ReturnLike from ControlFlowInterfaces.td
  let arguments = (outs Variadic<AnyType>:$operand);
}

#endif // HLIR_OPS
