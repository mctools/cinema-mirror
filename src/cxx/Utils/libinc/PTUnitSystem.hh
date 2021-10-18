#ifndef Prompt_PTUnitSystem_hh
#define Prompt_PTUnitSystem_hh

#include <cmath>
namespace Prompt {
  namespace Unit {
    constexpr double MeV = 1.;
    constexpr double TeV = 1.e6*MeV;
    constexpr double GeV = 1.e3*MeV;
    constexpr double keV = 1.e-3*MeV;
    constexpr double eV = 1.e-6*MeV;
    constexpr double meV = 1.e-9*MeV;

    constexpr double s = 1.;
    constexpr double ms = 1.e-3*s;
    constexpr double ns = 1.e-9*s;
    constexpr double ps = 1.e-12*s;
    constexpr double fs = 1.e-15*s;

    constexpr double mm = 1.;
    constexpr double cm = 0.1*mm;
    constexpr double m = 1e3*mm;
    constexpr double Aa = 1e-10*m;

    constexpr double g = 1.;
    constexpr double kg = 1.e3*g;
    constexpr double kelvin = 1.;
  }

  ///！ constants are directly obtained from NCrystal's source code
  constexpr double const_c  = 299792458*Unit::m/Unit::s ;// speed of light in Aa/s
  constexpr double const_dalton2kg =  1.660539040e-27*Unit::kg; // amu to kg (source: NIST/CODATA 2018)
  constexpr double const_dalton2eVc2 =  931494095.17*Unit::eV; // amu to eV/c^2 (source: NIST/CODATA 2018)
  constexpr double const_avogadro = 6.022140857e23; // mol^-1 (source: NIST/CODATA 2018)
  constexpr double const_dalton2gpermol = const_dalton2kg*const_avogadro; // dalton to gram/mol
  //NB: const_dalton2gmol is almost but not quite unity (cf. https://doi.org/10.1007/s00769-013-1004-9)

  constexpr double const_boltzmann = 8.6173303e-5*Unit::eV/Unit::kelvin;  // eV/K
  constexpr double const_neutron_mass_amu = 1.00866491588; // [amu]
  constexpr double const_proton_mass_amu = 1.007276466621; // [amu]
  constexpr double const_planck = 4.135667662e-15*Unit::eV*Unit::s ;//[eV*s]
  // constexpr double const_neutron_rest_mass = 939.56542052*Unit::MeV; // MeV (source: NIST/CODATA 2018)
  // constexpr double const_proton_rest_mass = 938.27208816*Unit::MeV; // MeV (source: NIST/CODATA 2018)

  //Derived values:
  constexpr double const_cc  = const_c*const_c;
  constexpr double const_neutron_mass_evc2 = const_neutron_mass_amu * const_dalton2eVc2 / (const_c*const_c);// [ eV/(Aa/s)^2 ]
  constexpr double const_proton_mass_evc2 = const_proton_mass_amu * const_dalton2eVc2 / (const_c*const_c);// [ eV/(Aa/s)^2 ]

  constexpr double const_ekin_2200m_s = 0.5 * const_neutron_mass_evc2 * 2200.0*Unit::m * 2200.0*Unit::m ; //neutron kinetic energy at 2200m/s [eV]
}
#endif
