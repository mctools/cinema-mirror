#include "GIDI.hpp"

#include "LUPI.hpp"
#include "MCGIDI.hpp"
#include "MCGIDI_sampling.hpp"
#include <limits>


#include "PTGIDIElasticModel.hh"


Prompt::GIDIElasticModel::GIDIElasticModel(const std::string &name, std::shared_ptr<MCGIDI::Protare> mcprotare,
            double temperature, double bias, double frac, double lowerlimt, double upperlimt)
:GIDIModel(name, mcprotare, temperature, bias, frac, lowerlimt, upperlimt), m_ncscatt(nullptr)
{
  unsigned numDigit = std::count_if(name.begin(), name.end(), 
      [](unsigned char c){ return std::isdigit(c); } );

  std::string element = name.substr(0, name.size()-numDigit);
  std::string cfgstr = "freegas::" + element + "/1gcm3/" + element + "_is_"+name+";temp="+std::to_string(temperature);
  m_ncscatt = std::make_shared<NCrystalScat>(cfgstr);

  std::cout << "GIDIElasticModel created " << cfgstr << std::endl;
}


void Prompt::GIDIElasticModel::generate(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir) const 
{

    // m_ncscatt->generate(ekin, dir, final_ekin, final_dir);

  if(ekin<2)
    m_ncscatt->generate(ekin, dir, final_ekin, final_dir);
  else
  {
    GIDIModel::generate(ekin, dir, final_ekin, final_dir);
  }

  // final_ekin=-1.;


}
