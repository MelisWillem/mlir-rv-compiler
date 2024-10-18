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
  let arguments = (ins AnyAttrOf<[I1Attr, I32Attr]>:$constant);

  let results = (outs AnyTypeOf<[I32, I1]>:$value);
}

def AllocaOp : HLIR_Op<"alloca">{
  let arguments = (ins 
    TypeAttr:$allocaType,
    StrAttr:$name
  );

  let results = (outs HLIR_PtrType:$ptr);
}

def Store : HLIR_Op<"Store">{
  let arguments = (ins 
    AnyType:$value,
    HLIR_PtrType:$ptr
  );
}

def Load : HLIR_Op<"Load">{
  let arguments = (ins 
    HLIR_PtrType:$ptr
  );
  let results =(outs AnyType:$value);
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

def IfOp : HLIR_Op<"If", [Terminator]>{
  let arguments = (ins I1:$cond);
  let successors = (successor 
    AnySuccessor:$trueDest,
    AnySuccessor:$falseDest);
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

def YieldOp : HLIR_Op<"Yield", [Terminator]>{ // maybe also add ReturnLike from ControlFlowInterfaces.td
  let arguments = (outs Variadic<AnyType>:$operand);
}

#endif // HLIR_OPS