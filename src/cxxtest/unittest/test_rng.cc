#include <chrono>
#include <iostream>

#include "PTRandCanonical.hh"

namespace pt = Prompt;

#include "../doctest.h"
TEST_CASE("SingletonRNG")
{
  auto &ref1 = pt::SingletonRNG::getInstance();
  printf("ref poniter %p\n", &ref1);

  auto &ref2 = pt::SingletonRNG::getInstance();
  printf("ref poniter %p\n", &ref2);

  CHECK(&ref1==&ref2);
}
