#include "PTMirrorPhysics.hh"
#include "PTUtils.hh"

Prompt::MirrorPhyiscs::MirrorPhyiscs(const std::string &cfgstring, double weightCut)
:Prompt::DiscreteModel(cfgstring, const_neutron_pgd,
                      std::numeric_limits<double>::min(), 10*Prompt::Unit::eV), m_wcut(weightCut)
{
  std::cout << "constructor mirror physics " << std::endl;
  //Eq. 5.1, McStas 2.3 components manual
  double m_i=4;
  double R0=1;
  double Qc=0.02;
  double alpha=6.49;
  double W=1./300;
  unsigned q_arr_size=1000;
  double q_max = 10;
  std::vector<double> q(q_arr_size);
  std::vector<double> r(q_arr_size);

  for(unsigned i=0;i<q_arr_size;i++)
  {
    q[i]=i*(q_max/q_arr_size);
    if(q[i]<Qc)
      r[i]=1.;
    else
      r[i]=0.5*R0*(1-tanh(( q[i]-m_i*Qc)/W))*(1-alpha*( q[i]-Qc));
  }
  m_table = std::make_shared<LookUpTable>(q, r, LookUpTable::kConst_Zero);
  std::cout << "constructor mirror physics completed" << std::endl;
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
  double angleCos = reflectionNor.angleCos(nDirInLab);

  double Q = neutronAngleCosine2Q(angleCos, ekin, ekin);
  scaleWeight =  m_table->get(Q);
  // std::cout << "Q " << Q << " scale " << scaleWeight << std::endl;
  if(m_wcut > scaleWeight)
    final_ekin = -1.0; //paprose kill

}
