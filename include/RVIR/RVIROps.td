#ifndef RVIR_OPS
#define RVIR_OPS

include "RVIR/RVIRTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"

// Physical registers only need 5 bit identifier, but virtual ones will use 32 bit identifier.
// reg x0-x31
class RVIR_Inst<string mnemonic, list<Trait> traits = []> 
: RVIR_Op<mnemonic, traits>{
    int opcode;
}

// Represents a register value, kind of a hack to fix a register in an instruction.
def ConstantRegOp : RVIR_Op<"Const", [Pure]>{
    let results = (outs RegisterType:$out);
}

def PopStack : RVIR_Op<"PopStack", [Pure]>{
    let summary = "Pop a value from the stack";
    let description = [{
        Pop a value from the stack.
    1. The stack pointer is in x2
    2. The stack pointer is decremented by index*4bytes
    3. The value is moved in the output register
    }];
    let arguments = (ins UI32Attr:$offset);
    let results = (outs RegisterType:$output);
}

def StoreStack : RVIR_Op<"StoreStack", [Pure]>{
    let summary = "Store a value to the stack";
    let description = [{
        Store a value to the stack.
    }];
    let arguments = (ins RegisterType:$input, UI32Attr:$offset);
    let results = (outs RegisterType:$output);
}

def RegValue : RVIR_Op<"RegVal", [Pure]>{
    let summary = [{
        Read out a register to a certain type. Used when lowering to RVIR.
    }];
    let arguments = (ins RegisterType:$input);
    let results = (outs AnyType);
}

def ValueReg : RVIR_Op<"ValReg", [Pure]>{
    let arguments = (ins AnyType:$input);
    let results = (outs RegisterType:$output);
}

// Integer register-immediate Instructions
class RVIR_RegisterImmediateInst <string mnemonic, list<Trait> traits = []>
    : RVIR_Inst<mnemonic, traits>{

    let arguments = (ins I32Attr:$imm, RegisterType:$rd);
    let results = (outs RegisterType:$rs1);
    let opcode = 19;

    int func3;
}

def RVIR_ADDI : RVIR_RegisterImmediateInst<"ADDI">{
    let summary = "Add an immediate to a register";
    let description = [{
        ADDI adds the sign-extended 12-bit immediate to register rs1. 
        Arithmetic overflow is ignored and the result is simply the low 
        XLEN bits of the result.ADDI rd,
      rs1, 0 is used to implement the MV rd, rs1 assembler pseudoinstruction.

      rd = rs1 + imm
}];
    let func3 = 0;
}

def RVIR_SLTI : RVIR_RegisterImmediateInst<"SLTI"> {
    let summary = "set less than immediate";
    let description = [{
        SLTI (set less than immediate) places the value 1 in register rd if register rs1 is less than the sign-
        extended immediate when both are treated as signed numbers, else 0 is written to rd.
    }];
    let func3 = 2;
}
def RVIR_SLTIU : RVIR_RegisterImmediateInst<"SLTIU">{
    let summary = "set less than immediate unsigned";
    let description = [{
        SLTIU is similar to SLTI but compares the values as unsigned numbers (i.e., the immediate is
        first sign-extended to XLEN bits then treated as an unsigned number). Note, SLTIU rd, rs1,
        1 sets rd to 1 if rs1 equals zero, otherwise sets rd to 0 (assembler pseudoinstruction SEQZ rd, rs).
    }];
    let func3 = 3;
}

// Logical operations
def RVIR_ANDI : RVIR_RegisterImmediateInst<"ANDI">{
    let summary = "logical AND between immediate and register";
    let description = [{
        register rs1 and the sign-extended 12-bit immediate and place the result in rd.
    }];
    let func3 = 7;
}
def RVIR_ORI : RVIR_RegisterImmediateInst<"ORI">{
    let summary = "logical OR between immediate and register";
    let description = [{
        Register rs1 and the sign-extended 12-bit immediate and place the result in rd.
    }];
    let func3 = 6;
}
def RVIR_XORI : RVIR_RegisterImmediateInst<"XORI">{
    let summary = "logical XOR between immediate and register";
    let description = [{
        Register rs1 and the sign-extended 12-bit immediate and place the result in rd.
        XORI rd, rs1, -1 performs
        a bitwise logical inversion of register rs1 (assembler pseudoinstruction NOT rd, rs)
    }];
    let func3 = 4;
}

class RVIR_RegisterImmediateInst_Shift<string mnemonic, list<Trait> traits = [] > 
    : RVIR_RegisterImmediateInst<mnemonic, traits> {
    int last7BitsImm;
    // TODO: put this as a constrain on the immediate.
  }
  def RVIR_SLLI : RVIR_RegisterImmediateInst_Shift<"SLLI"> {
    let summary = "shift logical left immediate";
    let description = [{
        SLLI is a logical left shift (zeros are shifted into the lower bits);

        The operand to be shifted is in rs1, and the shift amount is encoded
        in the lower 5 bits of the I-immediate field.
    }];
    let func3 = 1;
    let last7BitsImm = 0;
}
def RVIR_LRLI : RVIR_RegisterImmediateInst_Shift<"LRLI">{
    let summary = "shift right logical immediate";
    let description = [{
        SRLI is a logical right shift (zeros are shifted into the upper bits)

        The operand to be shifted is in rs1, and the shift  amount is encoded in
                        the lower 5 bits of the I -
                    immediate field.
    }];
    let func3 = 5;
    let last7BitsImm = 0;
}
def RVIR_SRAI : RVIR_RegisterImmediateInst_Shift<"SRAI"> {
    let summary = "shift right arithmetic immediate";
    let description = [{
        SRAI is an arithmetic right shift (the original sign bit is copied into
        the vacated upper bits).

        The operand to be shifted is in rs1, and the shift amount is encoded
        in the lower 5 bits of the I-immediate field.
    }];
    let func3 = 5;
    let last7BitsImm = 32;
}

class RVIR_UTypeInst <string mnemonic, list<Trait> traits = []>
    : RVIR_Inst<mnemonic, traits>{

    // imm: 31:12 -> 20 bits
    let arguments = (ins I32Attr:$imm, UI32Attr:$rd);
}
def RVIR_LUI : RVIR_UTypeInst<"LUI">{
    let summary = "load upper immediate";
    let description = [{
        LUI (load upper immediate) is used to build 32-bit constants and uses the U-type format. LUI
        places the U-immediate value in the top 20 bits of the destination register rd, filling in the lowest
        12 bits with zeros.
    }];
    let opcode = 55;
}
def RVIR_AUIPC : RVIR_UTypeInst<"AUIPC">{
    let summary = "add upper immediate to pc";
    let description = [{
        AUIPC (add upper immediate to pc) is used to build pc-relative addresses and uses the U-type
        format. AUIPC forms a 32-bit offset from the 20-bit U-immediate, filling in the lowest 12 bits with
        zeros, adds this offset to the address of the AUIPC instruction, then places the result in register
        rd.
    }];
    let opcode = 23;
}

class RVIR_RTypeInst <string mnemonic, list<Trait> traits = []>
    : RVIR_Inst<mnemonic, traits>{
    int func7;
    int func3;

    let arguments = (ins RegisterType:$rs1, RegisterType:$rs2);
    let results = (outs RegisterType:$rd);
    let opcode = 51;
}

// Integer Register-Register Operations
def RVIR_ADD : RVIR_RTypeInst<"ADD">{
    let summary = "addition of rs1 and rs2";
    let description = [{
        ADD performs1 the addition of rs1 and rs2.

        Overflows are ignored and the low XLEN bits of results are written to the destination rd.
    }];
    let func3 = 0;
    let func7 = 0;
}
def RVIR_SLT : RVIR_RTypeInst<"SLT">{
    let summary = "set less than signed";
    let description = [{
        writing 1 to rd if rs1 < rs2, 0 otherwise
    }];
    let func3 = 0;
    let func7 = 2;
}
def RVIR_SLTU : RVIR_RTypeInst<"SLTU">{
    let summary = "set less than unsigned";
    let description = [{
        writing 1 to rd if rs1 < rs2, 0 otherwise

        SLTU rd, x0, rs2 sets rd to 1 if rs2 is not equal to zero, otherwise sets rd to zero (assembler
        pseudoinstruction SNEZ rd, rs).
    }];
    let func3 = 3;
    let func7 = 0;
}

def RVIR_SLL : RVIR_RTypeInst<"SLL">{
    let summary = "Shift logical left";
    let description = [{
        Logical left shift on the value in register rs1 by the shift amount held in the lower 5 bits of register rs2.
    }];
    let func3 = 1;
    let func7 = 0;

}
def RVIR_SRL : RVIR_RTypeInst<"SRL">{
    let summary = "Shift logical right";
    let description = [{
        Logical right shift on the value in register rs1 by the shift amount held in the lower 5 bits of register rs2.
    }];
    let func3 = 5;
    let func7 = 0;
}

def RVIR_SUB : RVIR_RTypeInst<"SUB">{
    let summary = "subtraction of rs2 from rs1";
    let description = [{
        SUB performs the substrction of rs2 and rs1.

        Overflows are ignored and the low XLEN bits of results are written to the destination rd.
    }];
    let func3 = 0;
    let func7 = 32;
}
def RVIR_SRA : RVIR_RTypeInst<"SRA">{
    let summary = "Shift arithmetic right";
    let description = [{
        Arithmetic right shift on the value in register rs1 by the shift amount held in the lower 5 bits of register rs2.
    }];
    let func3 = 5;
    let func7 = 32;
}

// TODO: DO we need to represent nop? -> "NOP" == "ADDI x0, x0, 0"

// TODO:Plain unconditional jumps (assembler pseudoinstruction J) are encoded as a JAL with rd=x0.

// Control transfer instructions
def RVIR_JAL : RVIR_Inst<"JAL", [Terminator]>{
    // J is represented by JAL with rd=null
    let arguments = (UI32Attr : $rd);
    let successors = (successor AnySuccessor:$succ);
    let summary = "Jump and link";
    let description = [{
      The jump and link (JAL) instruction uses the J-type format, where the J-immediate encodes a
      signed offset in multiples of 2 bytes. The offset is sign-extended and added to the address of the
      jump instruction to form the jump target address. Jumps can therefore target a ±1 MiB range.
      JAL stores the address of the instruction following the jump (pc+4) into register rd. The standard
      software calling convention uses x1 as the return address register and x5 as an alternate link register.
    }];
    let opcode = 111;
}

def RVIR_JALR : RVIR_Inst<"JALR", [Terminator]>{
    // imm = 12 bit offset
    let arguments = (ins RegisterType:$rs1, RegisterType:$rd);
    let summary = "Jump and link register";
    let description = [{
       The indirect jump instruction JALR (jump and link register) uses the I-type encoding. The target
       address is obtained by adding the sign-extended 12-bit I-immediate to the register rs1, then setting
       the least-significant bit of the result to zero. The address of the instruction following the jump
       (pc+4) is written to register rd. Register x0 can be used as the destination if the result is not
       required.
    }];
    let opcode = 111;
    let successors = (successor AnySuccessor:$succ);
    int func3 = 0;
}

// Conditional branches
class RVIR_BRANCH <string mnemonic, list<Trait> traits = [Terminator]>
    : RVIR_Inst<mnemonic, traits>{
    // Also model the 2 registers here, that should be rs1 and rs2 as they can be
    // virtual registers...
    let arguments = (ins 
        RegisterType : $rs2,
        RegisterType : $rs1);
    int func3;
    let opcode = 99;

    let successors = (successor AnySuccessor:$succ);
}

def RVIR_BEQ : RVIR_BRANCH<"BEQ"> {
    let summary = "Branch if equal";
    let description = [{
        BEQ takes the branch if registers rs1 and rs2
        are equal respectively.
    }];
    let func3 = 0;
}

def RVIR_BNE : RVIR_BRANCH<"BNE">{
    let summary = "Branch if not equal";
    let description = [{
        BNE takes the branch if registers rs1 and rs2
        are unequal respectively.
    }];
    let func3 = 1;
}

def RVIR_BLT : RVIR_BRANCH<"BLT">{
    let summary = "Branch less then";
    let description = [{
        BLT takes the branch if rs1 is less than rs2, using
        signed comparison.
    }];
    let func3 = 4;
}
def RVIR_BLTU : RVIR_BRANCH<"BLTU">{
    let summary = "Branch less then unsigned comparison";
    let description = [{
        BLTU takes the branch if rs1 is less than rs2, using
        unsigned comparison.
    }];
    let func3 = 6;
}

def RVIR_BGE : RVIR_BRANCH<"BGE">{
    let summary = "Branch Greather or equal";
    let description = [{
        BGE takes the branch if rs1 is greater
        than or equal to rs2, using signed comparison respectively.
    }];
    let func3 = 5;
}
def RVIR_BGEU : RVIR_BRANCH<"BGEU">{
    let summary = "Branch Greather or equal unsigned comparison";
    let description = [{
        BGEU takes the branch if rs1 is greater
        than or equal to rs2, using unsigned comparison respectively.
    }];
    let func3 = 7;
}

// Load and store instructions
class RVIR_LOAD <string mnemonic, list<Trait> traits = []>
    : RVIR_Inst<mnemonic, traits>
{
    let opcode = 3;
    int func3; // width
    // offset of 12 bit
    let arguments = (ins I32Attr:$offset, UI32Attr:$rs1, UI32Attr:$rd);
}
def RVIR_LW : RVIR_LOAD<"LW">{
    let summary = "Load word";
    let description = [{
        The LW instruction loads a 32-bit value from memory into rd.

        The effective address is obtained by adding register rs1
        to the sign-extended 12-bit offset.
    }];
    let func3 = 2;
}
def RVIR_LH : RVIR_LOAD<"LH">{
    let summary = "Load Hex";
    let description = [{
        The LH instruction loads a 16-bit value from memory,
        then sign-extends to 32-bits before storing in rd.

        The effective address is obtained by adding register rs1
        to the sign-extended 12-bit offset.
    }];
    let func3 = 1;
}
def RVIR_LHU : RVIR_LOAD<"LHU">{
    let summary = "Load Hex UNSIGNED";
    let description = [{
        The LH instruction loads a 16-bit value from memory,
        then zero-extends to 32-bits before storing in rd.

        The effective address is obtained by adding register rs1
        to the sign-extended 12-bit offset.
    }];
    let func3 = 5;
}
def RVIR_LB : RVIR_LOAD<"LB">{
    let summary = "Load Byte";
    let description = [{
        The LW instruction loads a 8-bit value from memory into rd,
        then sign-extends to 32-bits before storing in rd.

        The effective address is obtained by adding register rs1
        to the sign-extended 12-bit offset.
    }];
    let func3 = 0;
}
def RVIR_LBU : RVIR_LOAD<"LBU">{
    let summary = "Load Byte Unsigned";
    let description = [{
        The LW instruction loads a 8-bit value from memory into rd,
        then zero-extends to 32-bits before storing in rd.

        The effective address is obtained by adding register rs1
        to the sign-extended 12-bit offset.
    }];
    let func3 = 4;
}

class RVIR_STORE <string mnemonic, list<Trait> traits = []>
    : RVIR_Inst<mnemonic, traits>
{
    let opcode = 35;
    int func3; // width
    // offset of 12 bit
    let arguments = (ins I32Attr:$offset, UI32Attr:$rs1, UI32Attr:$rs2);
}
def RVIR_SW : RVIR_STORE<"SW">{
    let summary = "Store word";
    let description = [{
       The SW store 32-bit, 16-bit of register rs2 to memory.

       The effective address is obtained by adding register rs1
       to the sign-extended 12-bit offset.
    }];
    let func3 = 2;
}
def RVIR_SH : RVIR_STORE<"SH">{
    let summary = "Store Hex";
    let description = [{
       The SH instruction stores 16-bit values from the low bits of register rs2 to memory.

       The effective address is obtained by adding register rs1
       to the sign-extended 12-bit offset.
    }];
    let func3 = 1;
}
def RVIR_SB : RVIR_STORE<"SB">{
    let summary = "Store Byte";
    let description = [{
       The SB instruction stores 8-bit values from the low bits of register rs2 to memory.

       The effective address is obtained by adding register rs1
       to the sign-extended 12-bit offset.
    }];
    let func3 = 0;
}

// not sure if the bool/I4 are good idea's but let's just roll with it
def bool : I<1>;
def I4 : I<4>;
def RVIR_FENCE : RVIR_Inst<"FENCE">{
    let summary = "order device I/O and memory accesses as viewed by other RISC- V harts and external devices or coprocessors";
    let description = [{
        The FENCE instruction is used to order device I/O and memory accesses as viewed by other RISC-
        V harts and external devices or coprocessors. Any combination of device input (I), device output
        (O), memory reads (R), and memory writes (W) may be ordered with respect to any combination
        of the same. Informally, no other RISC-V hart or external device can observe any operation in the
        successor set following a FENCE before any operation in the predecessor set preceding the FENCE.
    }];
    let arguments = (ins I4:$fm, bool:$PI, bool:$PO, bool:$PW, bool:$SI, bool:$SO, bool:$SR, bool:$SW, UI32Attr:$rs1, UI32Attr:$rd);
    int func3 = 0;
    let opcode = 15;
}

class RVIR_SYSTEM <string mnemonic, list<Trait> traits = []>
    : RVIR_Inst<mnemonic, traits>
{
    let opcode = 115;
    int func3;
    int func12;
    let arguments = (ins UI32Attr:$rs1, UI32Attr:$rd);
}
def RVIR_ECALL : RVIR_SYSTEM<"ECALL">{
    let summary = "service request";
    let description = [{
        The ECALL instruction is used to make a service request to the execution environment. The EEI
        will define how parameters for the service request are passed, but usually these will be in defined
        locations in the integer register file.
    }];
    int func3 = 0;
    int func12 = 0;
}
def RVIR_EBREAK : RVIR_SYSTEM<"EBREAK">{
    let summary = "return to debug enviroment";
    let description = [{
        The EBREAK instruction is used to return control to a debugging environment.
    }];
    int func3 = 0;
    int func12 = 1;
}

#endif // RVIR_OPS
