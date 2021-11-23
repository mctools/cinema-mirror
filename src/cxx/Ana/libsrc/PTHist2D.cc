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

  char buffer [500];
  int n =sprintf (buffer,
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.colors as colors\n"
    "data=np.loadtxt('%s')\n"
    "fig=plt.figure()\n"
    "ax = fig.add_subplot(111)\n"
    "pcm = ax.pcolormesh(data.T, cmap=plt.cm.jet,  norm=colors.LogNorm(vmin=data.max()*1e-10, vmax=data.max()), shading='auto')\n"
    "plt.show()\n", filename.c_str());

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
