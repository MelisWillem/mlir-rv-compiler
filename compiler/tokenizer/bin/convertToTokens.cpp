// Take input file, print out list of tokens.
// Used in the tests of the tokenizer.
#include <fstream>
#include <iostream>
#include <sstream>
#include <token/tokenizer.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Detected " << argc - 1 << " arguments. \n";
    std::cerr << "Only one argument is allowed -> path to the input file with "
                 "source code to be converted into tokens. \n";
  }

  std::string SourceFilename(argv[1]);

  std::ifstream ifs(SourceFilename);
  if (!ifs.good()) {
    std::cerr << "Can't read file with path=" << SourceFilename << "\n";
    return -1;
  }

  std::ostringstream oss;
  oss << ifs.rdbuf();
  std::string SourceCode = oss.str();

  tokenizer::Tokenizer tkz(SourceCode);
  auto success = tkz.Parse();

  std::cout << "Parsed " << tkz.getTokens().size() << " tokens. \n";

  for (auto t : tkz.getTokens()) {
    std::cout << "token=" << t << "\n";
  }

  if (!success) {
    std::cout << "Unable to tokenize source file(cursor="
              << tkz.getError()->cursor
              << " cursorBegin=" << tkz.getError()->cursorBegin
              << ") with error=" << tkz.getError()->message << "\n";
    return -1;
  }

  return 0;
}