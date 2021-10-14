#include "PTRandEngine.hh"

Prompt::RandEngine::RandEngine(uint64_t theseed)
{
  seed(theseed);
}

void Prompt::RandEngine::seed(uint64_t theseed)
{
  //Seed the state, using splitmix64 as recommended:
  m_s[0] = splitmix64(theseed);
  m_s[1] = splitmix64(theseed);

  //Mix up the state a little bit more, probably not really needed:
  for (unsigned i = 0; i<1000; i++)
    genUInt64();
}


uint64_t Prompt::RandEngine::genUInt64()
{
  uint64_t s0 = m_s[0];
  uint64_t s1 = m_s[1];
  uint64_t result = s0 + s1;
  s1 ^= s0;
  m_s[0] = ((s0 << 55) | (s0 >> 9)) ^ s1 ^ (s1 << 14);
  m_s[1] = (s1 << 36) | (s1 >> 28);
  return result;
}

Prompt::RandEngine::~RandEngine()
{
}

double Prompt::RandEngine::operator()()
{
  return genUInt64();
}

uint64_t Prompt::RandEngine::splitmix64(uint64_t& x)
{
  uint64_t z = (x += 0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}
