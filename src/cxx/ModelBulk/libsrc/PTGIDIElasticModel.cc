#include "GIDI.hpp"

#include "LUPI.hpp"
#include "MCGIDI.hpp"
#include "MCGIDI_sampling.hpp"
#include <limits>


#include "PTGIDIElasticModel.hh"

using Direction = Prompt::Vector;

Direction rotate_angle(
  Direction u, double mu, const double* phi)
{
  // Sample azimuthal angle in [0,2pi) if none provided
  double phi_;
  if (phi != nullptr) {
    phi_ = (*phi);
  } else {
    phi_ = 2.0 * M_PI * getRandNumber(nullptr);
  }

  // Precompute factors to save flops
  double sinphi = std::sin(phi_);
  double cosphi = std::cos(phi_);
  double a = std::sqrt(std::fmax(0., 1. - mu * mu));
  double b = std::sqrt(std::fmax(0., 1. - u.z() * u.z()));

  // Need to treat special case where sqrt(1 - w**2) is close to zero by
  // expanding about the v component rather than the w component
  if (b > 1e-10) {
    return {mu * u.x() + a * (u.x() * u.z() * cosphi - u.y() * sinphi) / b,
      mu * u.y() + a * (u.y() * u.z() * cosphi + u.x() * sinphi) / b,
      mu * u.z() - a * b * cosphi};
  } else {
    b = std::sqrt(1. - u.y() * u.y());
    return {mu * u.x() + a * (-u.x() * u.y() * sinphi + u.z() * cosphi) / b,
      mu * u.y() + a * b * sinphi,
      mu * u.z() - a * (u.y() * u.z() * sinphi + u.x() * cosphi) / b};
  }
}


Direction sample_cxs_target_velocity(
  double awr, double E, Direction u, double kT)
{
  double beta_vn = std::sqrt(awr * E / kT);
  double alpha = 1.0 / (1.0 + std::sqrt(M_PI) * beta_vn / 2.0);

  double beta_vt_sq;
  double mu;
  while (true) {
    // Sample two random numbers
    double r1 = getRandNumber(nullptr);
    double r2 = getRandNumber(nullptr);

    if (getRandNumber(nullptr) < alpha) {
      // With probability alpha, we sample the distribution p(y) =
      // y*e^(-y). This can be done with sampling scheme C45 from the Monte
      // Carlo sampler

      beta_vt_sq = -std::log(r1 * r2);

    } else {
      // With probability 1-alpha, we sample the distribution p(y) = y^2 *
      // e^(-y^2). This can be done with sampling scheme C61 from the Monte
      // Carlo sampler

      double c = std::cos(M_PI / 2.0 * getRandNumber(nullptr));
      beta_vt_sq = -std::log(r1) - std::log(r2) * c * c;
    }

    // Determine beta * vt
    double beta_vt = std::sqrt(beta_vt_sq);

    // Sample cosine of angle between neutron and target velocity
    mu = getRandNumber(nullptr)*2-1;

    // Determine rejection probability
    double accept_prob =
      std::sqrt(beta_vn * beta_vn + beta_vt_sq - 2 * beta_vn * beta_vt * mu) /
      (beta_vn + beta_vt);
    

    // Perform rejection sampling on vt and mu
    if (getRandNumber(nullptr) < accept_prob)
      break;
  }

  // Determine speed of target nucleus
  double vt = std::sqrt(beta_vt_sq * kT / awr);

  // Determine velocity vector of target nucleus based on neutron's velocity
  // and the sampled angle between them
  return rotate_angle(u, mu, nullptr)*vt;
}

void elastic_scatter(double awr, double kT, Prompt::Vector& dir, double& E)
{
  // get pointer to nuclide
  // const auto& nuc {data::nuclides[i_nuclide]};

  double vel = std::sqrt(E);

  // Neutron velocity in LAB
  Direction v_n = dir*vel;

  // Sample velocity of target nucleus
  Direction v_t {};
  v_t = sample_cxs_target_velocity(awr, E, dir, kT);


  // Velocity of center-of-mass
  Direction v_cm = (v_n + v_t*awr) / (awr + 1.0);

  // Transform to CM frame
  v_n -= v_cm;

  // Find speed of neutron in CM
  vel = v_n.mag();

  // // Sample scattering angle, checking if angle distribution is present (assume
  // // isotropic otherwise)
  // double mu_cm;
  // auto& d = rx.products_[0].distribution_[0];
  // auto d_ = dynamic_cast<UncorrelatedAngleEnergy*>(d.get());
  // if (!d_->angle().empty()) {
  //   mu_cm = d_->angle().sample(p.E(), p.current_seed());
  // } else {
  //   mu_cm = uniform_distribution(-1., 1., p.current_seed());
  // }
  double mu_cm = getRandNumber(nullptr)*2-1;

  // Determine direction cosines in CM
  Direction u_cm = v_n / vel;

  // Rotate neutron velocity vector to new angle -- note that the speed of the
  // neutron in CM does not change in elastic scattering. However, the speed
  // will change when we convert back to LAB
  v_n = rotate_angle(u_cm, mu_cm, nullptr)*vel;

  // Transform back to LAB frame
  v_n += v_cm;

  E = v_n.dot(v_n);
  vel = std::sqrt(E);

  // // compute cosine of scattering angle in LAB frame by taking dot product of
  // // neutron's pre- and post-collision angle
  // p.mu() = dir.dot(v_n) / vel;

  // Set energy and direction of particle in LAB frame
  dir = v_n / vel;

  // // Because of floating-point roundoff, it may be possible for mu_lab to be
  // // outside of the range [-1,1). In these cases, we just set mu_lab to exactly
  // // -1 or 1
  // if (std::abs(p.mu()) > 1.0)
  //   p.mu() = std::copysign(1.0, p.mu());
}



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
  // void elastic_scatter(double awr, double kT, Prompt::Vector& dir, double& E)

  // double awr = m_input->m_targetMass/m_input->m_projectileMass;
  // double awr= 235/1;
  // final_dir = dir;
  // final_ekin = ekin;
  // elastic_scatter(awr, 0.0253, final_dir, final_ekin);

    // m_ncscatt->generate(ekin, dir, final_ekin, final_dir);

  // if(ekin<m_mcprotare->URR_domainMin( )*2)
  //   m_ncscatt->generate(ekin, dir, final_ekin, final_dir);
  // else
  {
    GIDIModel::generate(ekin, dir, final_ekin, final_dir);
  }

  // final_dir.normalise();


}
