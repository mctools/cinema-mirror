////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  This file is part of Prompt (see https://gitlab.com/xxcai1/Prompt)        //
//                                                                            //
//  Copyright 2021-2022 Prompt developers                                     //
//                                                                            //
//  Licensed under the Apache License, Version 2.0 (the "License");           //
//  you may not use this file except in compliance with the License.          //
//  You may obtain a copy of the License at                                   //
//                                                                            //
//      http://www.apache.org/licenses/LICENSE-2.0                            //
//                                                                            //
//  Unless required by applicable law or agreed to in writing, software       //
//  distributed under the License is distributed on an "AS IS" BASIS,         //
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  //
//  See the License for the specific language governing permissions and       //
//  limitations under the License.                                            //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "PTHist1D.hh"
#include "PTMath.hh"
#include "PTBinaryWR.hh"
#include <typeinfo>
#include "PTUtils.hh"
#include "PTRandCanonical.hh"

Prompt::Hist1D::Hist1D(const std::string &name, double xmin, double xmax, unsigned nbins, bool linear)
:HistBase(name, nbins), m_binfactor(0), m_linear(linear), m_logxmin(0)
{
  m_xmin=xmin, m_xmax=xmax, m_nbins=nbins;
  if(linear) {
    if(xmin==xmax)
      PROMPT_THROW(BadInput, "xmin and xman can not be equal");
    m_binfactor=nbins/(xmax-xmin);
  }
  else {
    if(xmin<=0 || xmax<=0)
      PROMPT_THROW(BadInput, "xmin and xman must be positive");
    m_binfactor=nbins/(log10(xmax)-log10(xmin));
    m_logxmin=log10(m_xmin);
  }
}

Prompt::Hist1D::~Hist1D()
{
}

std::vector<double> Prompt::Hist1D::getEdge() const
{
  if(m_linear)
    return linspace(m_xmin, m_xmax, m_nbins+1);
  else
    return logspace(log10(m_xmin), log10(m_xmax), m_nbins+1);
}

void Prompt::Hist1D::save(const std::string &filename) const
{
  auto seed = Singleton<SingletonPTRand>::getInstance().getSeed();
  auto bwr = BinaryWrite(filename+"_seed"+std::to_string(seed));

  double intergral(getIntegral()), overflow(getOverflow()), underflow(getUnderflow());
  bwr.addHeaderComment(getTypeName(typeid(this)).c_str());
  bwr.addHeaderComment(("total hit: " + std::to_string(getTotalHit())).c_str());

  bwr.addHeaderComment(("integral weight: " + std::to_string(intergral )).c_str());
  bwr.addHeaderComment(("accumulated weight: " + std::to_string(intergral-overflow-underflow)).c_str());
  bwr.addHeaderComment(("overflow weight: " + std::to_string(overflow )).c_str());
  bwr.addHeaderComment(("underflow weight: " + std::to_string(underflow)).c_str());

  bwr.addHeaderData("overflow", &overflow, {1}, Prompt::NumpyWriter::NPDataType::f8);
  bwr.addHeaderData("underflow", &underflow, {1}, Prompt::NumpyWriter::NPDataType::f8);

  bwr.addHeaderData("content", m_data.data(), {m_nbins}, Prompt::NumpyWriter::NPDataType::f8);
  bwr.addHeaderData("hit", m_hit.data(), {m_nbins}, Prompt::NumpyWriter::NPDataType::f8);
  bwr.addHeaderData("edge", getEdge().data(), {m_nbins+1}, Prompt::NumpyWriter::NPDataType::f8);

  char buffer [1000];
  int n =sprintf (buffer,
    "import numpy as np\nfrom Cinema.Prompt import PromptFileReader\n"
    "import matplotlib.pyplot as plt\n"
    "f = PromptFileReader('%s_seed%ld.mcpl.gz')\n"
    "x=f.getData('edge')\n"
    "y=f.getData('content')\n"
    "plt.%s(x[:-1],y/np.diff(x), label=f'integral={y.sum()}')\n"
    "plt.grid()\n"
    "plt.legend()\n"
    "plt.show()\n", filename.c_str(), seed, m_linear? "plot":"loglog");

  std::ofstream outfile(filename+"_view.py");
  outfile << buffer;
  outfile.close();
}

//Normal filling:
void Prompt::Hist1D::fill(double val)
{
  fill(val, 1.);
}

void Prompt::Hist1D::fill(double val, double w)
{
  std::lock_guard<std::mutex> guard(m_hist_mutex);

  m_sumW+=w;
  if(val<m_xmin) {
    m_underflow += w;
    return;
  }
  else if(val>m_xmax) {
    m_overflow += w;
    return;
  }

  unsigned i = m_linear ? floor((val-m_xmin)*m_binfactor) : floor((log10(val)-m_logxmin)*m_binfactor) ;
  m_data[i] += w;
  m_hit[i] += 1;
}
