//===- RVIRPasses.h - RVIR passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef RVIR_RVIRPASSES_H
#define RVIR_RVIRPASSES_H

#include "RVIR/RVIRDialect.h"
#include "RVIR/RVIROps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace rvir {
#define GEN_PASS_DECL
#include "RVIR/RVIRPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "RVIR/RVIRPasses.h.inc"
} // namespace rvir
} // namespace mlir

#endif
