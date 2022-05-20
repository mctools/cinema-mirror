#include "Hdf5Trj.hh"
#include "Fourier.hh"
#include <vector>
#include <fstream>
#include <iostream>
#include <map>
#include <cmath>
#include "Utils.hh"
#include <getopt.h> //long_options
#include <sstream>
#include <algorithm>

std::vector<std::string> split(const std::string& text, char delimiter)
{
  std::vector<std::string> words{};

  std::stringstream sstream(text);
  std::string word;
  while (std::getline(sstream, word, delimiter))
      words.push_back(word);
  return words;
}


int main(int argc, char *argv[])
{
  Data<double> Q;
  std::vector<double> scatt_len{};

  std::string fname(""), saveAs("");
  bool vdos_only(false), coherent(false);
  std::map<unsigned, std::string> taskMap;
  int sqsize=0;

  int opt_char;
  int option_index;
  static struct option long_options[] = {
    {"input", required_argument, 0, 'i'},
    {"output", required_argument, 0, 'o'},
    {"coherent", no_argument, 0, 'c'},
    {"vdos_only", no_argument, 0, 'v'},
    {"incoh_qvec", required_argument, 0, 'k'},
    {"coh_qvec", required_argument, 0, 'q'},
    {"help",        no_argument,       0, 'h'},
    {NULL,          0,                 0, 0}
  };

  while((opt_char = getopt_long(argc, argv, "i:o:k:q:vht:l:c",
            long_options, &option_index)) != -1) {
  switch (opt_char) {
      case 'i':
      {
        fname=std::string(optarg);
        printf("input file: %s\n", fname.c_str());
        break;
      }
      case 'o':
      {
        saveAs=std::string(optarg);
        printf("output file: %s\n", saveAs.c_str());
        break;
      }
      case 'l':
      {
        auto scl=split(std::string(optarg), ',');
        for(auto v:scl)
        {
          scatt_len.push_back(atof(v.c_str()));
        }
        break;
      }
      case 'c':
      {
        coherent=true;
        break;
      }
      case 'k':
      {
        auto incoh_qvec=split(std::string(optarg), ',');

        if(incoh_qvec[0].compare("log") && incoh_qvec[0].compare("lin") && incoh_qvec[0].compare("qlist") )
        {
           printf("The first parameter of incho_qvec should be \"log\", \"lin\" or \"qlist\", but is \"%s\"\n", incoh_qvec[0].c_str());
           exit(1);
        }
        if(!incoh_qvec[0].compare("log")  )
        {
          if(incoh_qvec.size()!=4)
          {
            printf("incoh_qvec should be defined with 4 parameters\n");
            exit(1);
          }
          Q.vec=logspace(std::stod(incoh_qvec[1]), std::stod(incoh_qvec[2]), std::stod(incoh_qvec[3]));
        }
        else if(!incoh_qvec[0].compare("lin")  )
        {
          if(incoh_qvec.size()!=4)
          {
            printf("incoh_qvec should be defined with 4 parameters\n");
            exit(1);
          }
          Q.vec=linspace(std::stod(incoh_qvec[1]), std::stod(incoh_qvec[2]), std::stod(incoh_qvec[3]));
        }
        else if(!incoh_qvec[0].compare("qlist")  )
        {
          Q.vec.clear();
          for(unsigned i=1;i<incoh_qvec.size();i++)
            Q.vec.push_back(std::stod(incoh_qvec[i]));
        }

        Q.dim.resize(1);
        Q.dim[0]=Q.vec.size();
        if(Q.vec[0]<0.)
        {
          printf("Q must be positive\n");
          exit(1);
        }
        break;
      }
      case 't':
      {
        auto w=split(std::string(optarg), ';');
        for(auto &substr : w)
        {
          auto s=split(substr, ',');
          taskMap[atoi(s[0].c_str())]=s[1];
        }
        break;
      }
      case 'v':
      {
        vdos_only=true;
        printf("vDOS only\n");
        break;
      }
      case 'q':
      {
        sqsize=atoi(optarg);
        break;
      }
      case 'h':
      {
        printf("error optopt: %c\n", optopt);
        printf("error opterr: %d\n", opterr);
        break;
      }
    }
  }

  Hdf5Trj trj(fname);

  //Cache atomic trajectories into memory.
  //Work around of slow hyperslab access of hdf5.
  if(!taskMap.empty() || coherent)
  {
    if(vdos_only)
    {
      Q.clear();
      printf("Calculates vDos only without incoherent scattering function:\n");
    }
    else
    {
      printf("The Q vector for the incoherent scattering function is:\n");
      for(auto v: Q.vec)
        printf("%g ", v);
      printf("\n");
    }

    trj.cacheAtomTrj();
  }

  Data<double>::createH5File(saveAs);

  double qRes = trj.getQResolution();
  Data<double> vdos;
  Data<double> sqw;
  Data<double> omega;

  if(!taskMap.empty())
  {
    for (const auto& kv : taskMap)
    {
      std::cout << "starting task " << kv.second << std::endl;

      trj.vDosSqw(kv.first, vdos, omega, Q.vec, sqw); //heavy calculation here

      double freScale =trj.getDeltaT()/(2*M_PI);

      double weight = trapz(vdos.vec, omega.vec);
      vdos.vec = vdos.vec*(1./weight);
      sqw.vec = sqw.vec*freScale;

      vdos.moveToH5( "inc_vdos_"+kv.second);
      omega.moveToH5( "inc_omega_"+kv.second);

      if(!vdos_only)
      {
        Data<double> tvec;
        tvec.dim.resize(1);
        tvec.dim[0]=trj.getNumFrame();
        tvec.vec.resize(tvec.dim[0]);
        double tbasis=trj.getDeltaT();
        for(unsigned i=0;i<tvec.dim[0];++i)
        {
          tvec.vec[i]=(i*tbasis);
        }
        tvec.moveToH5( "tVec_"+kv.second);
        sqw.moveToH5( "inc_sqw_"+kv.second);
        Q.saveAsH5( "qVec_"+kv.second);
      }
    }
  }

  if(coherent)
  {
    Data<double> cosqw;
    trj.coherentDynamicsFactor(sqsize, cosqw, scatt_len);
    cosqw.moveToH5( "coh_sqw");

    double freScale =trj.getDeltaT()/(2*M_PI);
    auto omega = trj.unweightedFrequency();
    omega.vec = omega.vec*(1./freScale);
    omega.moveToH5( "coh_omega");

  }

  //clear cache
  if(!taskMap.empty() || coherent)
  {
    std::cout << "clearing cache" << std::endl;
    trj.clearCache();
  }

  if(sqsize)
  {
    Data<double> sq_s;
    if(scatt_len.empty())
    {
      for(unsigned i=0;i<trj.getNumAtomPerMolecule();++i)
      {
        scatt_len.push_back(1.);
      }
    }
    trj.structFactSqw( sqsize , sq_s, scatt_len);
    sq_s.moveToH5( "sq_s");
    Data<double> sq_q;
    for(unsigned i=0;i<sqsize;i++)
    {
      sq_q.vec.push_back(trj.getQResolution()*(i+1));
    }
    sq_q.dim.push_back(sqsize);
    sq_q.moveToH5( "sq_q");
  }

  Data<double>::closeH5File();

}
