#include "PTPrimaryGun.hh"


Prompt::Particle Prompt::PrimaryGun::generate()
{
  sampleEnergy(m_ekin);
  samplePosDir(m_pos, m_dir);
  m_eventid++;
  return *this;
}


// def sample(self): #maxwellian with a tail
//   KiEn=0
//   if(self.rand() < 0.895230796325):
//     wl=0.
//     while(wl < self.threshold_wavelength):
//       wl=self.wavelength_T/ math.sqrt(-math.log( self.rand() * self.rand() ) )
//     KiEn=Utils.NeutronMath.neutronWavelengthToEKin(wl)
//   else:
//     r = self.rand()
//     KiEn=math.pow(self.threshold_energy,(1-r)) * math.pow(1*Units.keV , r)
//   return KiEn
