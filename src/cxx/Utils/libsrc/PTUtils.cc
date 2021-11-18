#include "PTUtils.hh"

std::vector<std::string> Prompt::split(const std::string& text, char delimiter)
{
  std::vector<std::string> words;
  std::stringstream sstream(text);
  std::string word;
  while (std::getline(sstream, word, delimiter))
      words.push_back(word);

  return words;
}

Prompt::Vector Prompt::string2vec(const std::string& text, char delimiter)
{
  auto subs = split(text, delimiter);
  if(subs.size()!=3)
    PROMPT_THROW2(BadInput, "string2vec " << text);
  return Vector{std::stod(subs[0]), std::stod(subs[1]),std::stod(subs[2]) };
}
