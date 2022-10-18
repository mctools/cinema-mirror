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

#include "PTHist2D.hh"

Prompt::Hist2D::Hist2D(double xmin, double xmax, unsigned xnbins,
                       double ymin, double ymax, unsigned ynbins)
:HistBase(xnbins*ynbins), m_xbinfactor(xnbins/(xmax-xmin)),
m_ybinfactor(ynbins/(ymax-ymin))
{
  m_xmin=xmin, m_xmax=xmax, m_xnbins=xnbins;
  m_ymin=ymin, m_ymax=ymax, m_ynbins=ynbins;
  m_nbins = m_xnbins * m_ynbins;

  if(xnbins*ynbins==0)
    PROMPT_THROW(BadInput, "bin size is zero");

  if(xmax<=xmin || ymax<=ymin)
    PROMPT_THROW(BadInput, "max<min");

}

Prompt::Hist2D::~Hist2D()
{
}

void Prompt::Hist2D::operator+=(const Hist2D& hist)
{
  auto data=hist.getRaw();
  if(data.size()!=m_data.size())
    PROMPT_THROW(BadInput, "operator+= hist with different data size");
  std::lock_guard<std::mutex> guard(m_hist_mutex);
  for(unsigned i=0;i<data.size();++i)
    m_data[i]+=data[i];
}

#include<iostream>
#include<fstream>
void Prompt::Hist2D::save(const std::string &filename) const
{
  std::cout << "total count " << getTotalHit() << std::endl;
  std::ofstream ofs;
  ofs.open(filename, std::ios::out);

  for(uint32_t i=0;i<m_xnbins;i++)
  {
    for(uint32_t j=0;j<m_ynbins;j++)
    {
      ofs << m_data[i*m_ynbins + j] << " ";
    }
    ofs << "\n";
  }
  ofs.close();

  ofs.open(filename+"_hit", std::ios::out);

  for(uint32_t i=0;i<m_xnbins;i++)
  {
    for(uint32_t j=0;j<m_ynbins;j++)
    {
      ofs << m_hit[i*m_ynbins + j] << " ";
    }
    ofs << "\n";
  }
  ofs.close();

  char buffer [1000];
  //fixme: add xy to dimansion
  int n =sprintf (buffer,
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.colors as colors\n"
    "data=np.loadtxt('%s')\n"
    "fig=plt.figure()\n"
    "ax = fig.add_subplot(111)\n"
    "pcm = ax.pcolormesh(data.T, cmap=plt.cm.jet,shading='auto')\n"
    "#pcm = ax.pcolormesh(data.T, cmap=plt.cm.jet, norm=colors.LogNorm(vmin=data.max()*1e-10, vmax=data.max()), shading='auto')\n"
    "fig.colorbar(pcm, ax=ax)\n"
    "count=np.loadtxt('%s')\n"
    "count=count.sum()-count.max()\n"
    "integral= data.sum()\n"
    "plt.title(f'Integral {integral}, count {count}')\n"
    "plt.show()\n", filename.c_str(), (filename+"_hit").c_str());

  std::ofstream outfile(filename+"_view.py");
  outfile << buffer;
  outfile.close();

}

//Normal filling:
void Prompt::Hist2D::fill(double xval, double yval)
{
  fill(xval, yval, 1.);
}

void Prompt::Hist2D::fill(double xval, double yval, double w)
{
  std::lock_guard<std::mutex> guard(m_hist_mutex);
  fill_unguard(xval, yval, w);
}

void Prompt::Hist2D::merge(const Prompt::HistBase &hist)
{
  auto &ref = dynamic_cast<const Prompt::Hist2D&>(hist);

  if(m_xbinfactor!=ref.m_xbinfactor)
    PROMPT_THROW2(CalcError, "m_xbinfactor " << m_xbinfactor << " is different with the m_xbinfactor of another histogram " << ref.m_xbinfactor);

  if(m_ybinfactor!=ref.m_ybinfactor)
    PROMPT_THROW2(CalcError, "m_ybinfactor " << m_ybinfactor << " is different with the m_ybinfactor of another histogram " << ref.m_ybinfactor);

  if(m_ymin!=ref.m_ymin)
    PROMPT_THROW2(CalcError, "m_ymin " << m_ymin << " is different with the m_ymin of another histogram " << ref.m_ymin);

  if(m_ymax!=ref.m_ymax)
    PROMPT_THROW2(CalcError, "m_ymax " << m_ymax << " is different with the m_ymax of another histogram " << ref.m_ymax);

  if(m_xnbins!=ref.m_xnbins)
    PROMPT_THROW2(CalcError, "m_xnbins " << m_xnbins << " is different with the m_xnbins of another histogram " << ref.m_xnbins);

  if(m_ynbins!=ref.m_ynbins)
    PROMPT_THROW2(CalcError, "m_ynbins " << m_ynbins << " is different with the m_ynbins of another histogram " << ref.m_ynbins);

  HistBase::merge(hist);

}
