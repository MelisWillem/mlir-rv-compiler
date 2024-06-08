//===- RVIRDialect.cpp - RVIR dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RVIR/RVIRDialect.h"
#include "RVIR/RVIROps.h"
#include "RVIR/RVIRTypes.h"

using namespace mlir;
using namespace mlir::rvir;

#include "RVIR/RVIROpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// RVIR dialect.
//===----------------------------------------------------------------------===//

void RVIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "RVIR/RVIROps.cpp.inc"
      >();
  registerTypes();
}
