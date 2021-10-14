#include "PTVector.hh"
#include <iostream>

std::ostream& Prompt::operator << (std::ostream &o, const Prompt::Vector& vec)
{
  return o<<"{ " << vec.x() <<", " << vec.y() << ", " << vec.z() << " }";
}

void Prompt::Vector::print() const
{
  std::cout << *this <<std::endl;
}

void Prompt::Vector::setMag(double f)
{
  if (f<0)
    PROMPT_THROW(BadInput,"NCVector::setMag(): Can't set negative magnitude.");
  double themag2 = mag2();
  if (!themag2)
    PROMPT_THROW(BadInput,"NCVector::setMag(): Can't scale null-vector.");
  double ff = f / sqrt(themag2);
  m_x *= ff;
  m_y *= ff;
  m_z *= ff;
}
