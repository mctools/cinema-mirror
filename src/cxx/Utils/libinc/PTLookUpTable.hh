#ifndef Prompt_LookUpTable_hh
#define Prompt_LookUpTable_hh

#include <functional>
#include <vector>
#include "PromptCore.hh"

namespace Prompt {
  //A table for linear interplation of a curve.
  //Integration of curve can also be applied.
  class LookUpTable {
  public:
    //extrapolation methods below lower and beyond upper bounds
    enum Extrapolate {
      kConst_Zero,
      kZero_Zero,
      kZero_Const,
      kOverX_Zero,
      kOverSqrtX_Zero,
      kOverSqrtX_OverSqrtX,
      kConst_OverSqrtX,
    };
  public:
    LookUpTable(const std::vector<double>& x, const std::vector<double>& f, Extrapolate extrap=Extrapolate::kZero_Zero);
    LookUpTable();
    virtual ~LookUpTable();
    bool empty() const;
    void sanityCheck() const;
    double get(double x) const;
    void print() const;
    virtual double integrate(double lower_x, double upper_x);

  private:
    void init();
    std::vector<double> m_x, m_f;
    std::function<double(double)>  m_func_extrapLower, m_func_extrapUpper;
    double extrapZero(double );
    double extrapConstUpper(double );
    double extrapConstLower(double );
    double extrapOverSqrtXLower(double x);
    double extrapOverSqrtXUpper(double x);
    double extrapOverXLower(double x);
    double extrapOverXUpper(double x);
  };
  #include "PTLookUpTable.icc"
}

#endif
