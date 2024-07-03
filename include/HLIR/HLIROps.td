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

def ConstantOp : HLIR_Op<"Constant">{
  let arguments = (ins I32Attr:$valueAttr);

  let results = (outs I32:$value);
}

def CallOp : HLIR_Op<"Call">{
  let arguments = (ins FlatSymbolRefAttr:$sym_name);
}

def FuncOp : HLIR_Op<"func"> {
  let summary = "function operation";
  let description = [{
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);
}

def CompareOp : HLIR_Op<"Compare">{
  let arguments = (ins CmpTypeAttr:$type, I32:$left, I32:$right);

  let results = (outs I1:$res);
}

def IfOp : HLIR_Op<"If">{
  let arguments = (ins I1:$cond);

  let regions = (region AnyRegion:$body);
}

// How to connect if and else?(should we even do it?)
def ElsOp : HLIR_Op<"else">{
  let regions = (region AnyRegion:$body);
}

def ReturnOp : HLIR_Op<"Return", [Terminator]>{ // maybe also add ReturnLike from ControlFlowInterfaces.td
  let arguments = (outs Optional<AnyType>:$operand);
}

def ForOp : HLIR_Op<"For">{
  let regions = (region AnyRegion:$body);
}

#endif // HLIR_OPS