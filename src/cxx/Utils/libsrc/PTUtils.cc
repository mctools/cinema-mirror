#include "PTUtils.hh"

void Prompt::split(const std::string& text, char delimiter, std::vector<std::string> &words)
{
  words.clear();
  std::stringstream sstream(text);
  std::string word;
  while (std::getline(sstream, word, delimiter))
      words.push_back(word);
}
