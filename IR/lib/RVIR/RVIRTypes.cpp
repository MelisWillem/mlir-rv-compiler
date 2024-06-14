//===- RVIRTypes.cpp - RVIR dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RVIR/RVIRTypes.h"

#include "RVIR/RVIRDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir::rvir;

#define GET_TYPEDEF_CLASSES
#include "RVIR/RVIROpsTypes.cpp.inc"

void RVIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "RVIR/RVIROpsTypes.cpp.inc"
      >();
}
