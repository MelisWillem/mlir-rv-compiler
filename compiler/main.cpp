#include "parser/parser.h"
#include "token/tokenizer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <iostream>
#include <sstream>

using namespace llvm;

cl::list<std::string> SourceFilenames("i", cl::desc("Specify source filename"),
                                      cl::value_desc("filename"), cl::Required);
cl::opt<bool> HIL("HIL", cl::desc("Output HIL"), cl::init(false));
cl::opt<std::string> OutputFile("o", cl::desc("Specify output filename"),
                                cl::value_desc("output"), cl::Required);
cl::opt<bool> debug_parser("debug_parser",
                           cl::desc("print debug output for the parser"),
                           cl::init(false));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  if (SourceFilenames.empty()) {
    std::cerr << "Provide input file location. \n";
  }

  for (auto &SourceFilename : SourceFilenames) {
    std::cout << "reading " << SourceFilename << "\n";
    std::ifstream ifs(SourceFilename);
    if (!ifs.good()) {
      std::cout << "Can't read file with path=" << SourceFilename << "\n";
      return -1;
    }

    std::ostringstream oss;
    oss << ifs.rdbuf();
    std::string SourceCode = oss.str();
    tokenizer::Tokenizer tkz(SourceCode);
    if (!tkz.Parse()) {
      std::cout << "Failed to parse file=" << SourceFilename
                << tkz.getError()->ToString() << "\n";
      return -1;
    }
    mlir::MLIRContext context;
    std::cout << "Found " << tkz.getTokens().size() << " tokens \n";
    parser::Parser parser(tkz.getTokens(), &context, SourceFilename, SourceCode,
                          debug_parser);
    auto module = parser.Parse();

    if (!module.has_value()) {
      std::cout << "failed to parse \n";
    }

    if (HIL) {
      std::error_code err;
      llvm::raw_fd_ostream output(OutputFile, err);
      module->print(output);
    } else {
      std::cout << "parsed file, set OutputHIL to get the HIL \n";
    }
  }

  return 0;
}