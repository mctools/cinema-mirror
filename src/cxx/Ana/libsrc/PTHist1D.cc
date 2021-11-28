#include "PTHist1D.hh"
#include "PTMath.hh"

Prompt::Hist1D::Hist1D(double xmin, double xmax, unsigned nbins, bool linear)
:HistBase(nbins), m_binfactor(0), m_linear(linear), m_logxmin(0)
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

#include "PTRandCanonical.hh"

void Prompt::Hist1D::save(const std::string &filename) const
{
  auto seed = Singleton<SingletonPTRand>::getInstance().getSeed();
  NumpyWriter nvt;
  nvt.writeNumpyFile(filename+"_seed"+std::to_string(seed)+"_content.npy", m_data, NumpyWriter::data_type::f8,
                   std::vector<uint64_t>{m_nbins});

  nvt.writeNumpyFile(filename+"_seed"+std::to_string(seed)+"_edge.npy", getEdge(), NumpyWriter::data_type::f8,
                   std::vector<uint64_t>{m_nbins+1});

  char buffer [500];
  int n =sprintf (buffer,
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "x=np.load('%s_seed%ld_edge.npy')\n"
    "y=np.load('%s_seed%ld_content.npy')\n"
    "plt.%s(x[:-1],y/np.diff(x), label=f'integral={y.sum()}')\n"
    "plt.grid()\n"
    "plt.legend()\n"
    "plt.show()\n", filename.c_str(), seed, filename.c_str(), seed, m_linear? "plot":"loglog");

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
