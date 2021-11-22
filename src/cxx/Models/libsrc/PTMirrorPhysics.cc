#include "PTMirrorPhysics.hh"
#include "PTUtils.hh"

Prompt::MirrorPhyiscs::MirrorPhyiscs(const std::string &cfgstring)
:Prompt::DiscreteModel(cfgstring, const_neutron_pgd,
                      std::numeric_limits<double>::min(), 10*Prompt::Unit::eV)
{
  //Eq. 5.1, McStas 2.3 components manual
  double m_i=4;
  double R0=1;
  double Qc=0.02;
  double alpha=6.49;
  double W=1./300;
  unsigned q_arr_size=1000;
  double q_max = 0.13;
  double *q = new double[q_arr_size];
  double *r = new double[q_arr_size];
  for(unsigned i=0;i<q_arr_size;i++)
  {
    q[i]=q_max/q_arr_size*i;
    if(q[i]<Qc)
      r[i]=1.;
    else
      r[i]=0.5*R0*(1-tanh(( q[i]-m_i*Qc)/W))*(1-alpha*( q[i]-Qc));
  }

  // m_reflectivity = new G4PhysicsOrderedFreeVector(q,r,q_arr_size);
  delete[] q;
  delete[] r;
}


Prompt::MirrorPhyiscs::~MirrorPhyiscs()
{}

double Prompt::MirrorPhyiscs::getCrossSection(double ekin) const
{
  return 0.;
}

double Prompt::MirrorPhyiscs::getCrossSection(double ekin, const Vector &dir) const
{
  return 0.;
}

void Prompt::MirrorPhyiscs::generate(double ekin, const Vector &nDirInLab, double &final_ekin, Vector &reflectionNor, double &scaleWeight) const
{
  final_ekin=ekin;
  reflectionNor = nDirInLab - reflectionNor*(2*(nDirInLab.dot(reflectionNor)));


}
