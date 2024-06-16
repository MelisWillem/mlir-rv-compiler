#include "llvm/Support/CommandLine.h"
#include <fstream>
#include <iostream>
#include <sstream>

using namespace llvm;

cl::list<std::string>
    SourceFilenames("i", cl::desc("Specify source filename"),
                    cl::value_desc("filename"), cl::Required);
cl::opt<bool> OutputHIL("printHIL", cl::desc("Print out the HIL after parsing"),
                        cl::value_desc("printHIL"), cl::init(false));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  for (auto &SourceFilename : SourceFilenames) {
    std::ifstream ifs(SourceFilename);
    if (!ifs.good()) {
        std::cout << "Can't read file with path=" << SourceFilename << "\n";
        return -1;
    }

    std::ostringstream oss;
    oss << ifs.rdbuf();
    std::string SourceFile = oss.str();
    std::cout << SourceFile << "\n";
  }

  return 0;
}