#ifndef Data_hh
#define Data_hh

#include <string>
#include <stdexcept>
#include <ostream>
#include <iostream>
#include <vector>
#include <chrono>
#include "hdf5.h"

template <class T>
struct Data {
  std::string dataName;
  std::vector<T> vec;
  std::vector<long long unsigned int> dim;
  inline static hid_t fh5;

  ~Data()
  {
  }

  static void closeH5File()
  {
    H5Fclose(fh5);
  }

  static void createH5File(const std::string& fname)
  {
    fh5 = H5Fcreate (fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  }


  void moveToH5(const std::string& dsname)
  {
    saveAsH5(dsname);
    clear();
  }

  void saveAsH5(const std::string& dsname) //fixme: double only
  {
    dataName=dsname; //set name so sanityCheck can feedback which operation is wrong
    sanityCheck();
    hid_t       space, dset;
    herr_t      status;

    unsigned rank = dim.size();
    hsize_t *dims = new hsize_t[rank];
    for(unsigned i=0;i<rank;i++)
    {
      dims[i]=dim[i];
    }

    space = H5Screate_simple (rank, dims, NULL);
    dset = H5Dcreate2 (fh5, dsname.c_str(), H5T_NATIVE_DOUBLE, space, H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    status = H5Dwrite (dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, vec.data());
    status = H5Dclose (dset);
    status = H5Sclose (space);

    delete [] dims;
  }

  void sanityCheck()
  {
    size_t totSize = 1;
    for(auto v: dim)
      totSize *= v;
    if(totSize-vec.size())
      throw std::runtime_error ( dataName +" is in wrong dimensions");
  }

  void clear()
  {
    dim.resize(1);
    dim[0]=0;
    std::vector<T> temp;
    vec.swap(temp);
  }

  void print(std::string info="", long long unsigned printSize=10)
  {
    std::cout << "********\n";
    info.empty() ? std::cout << dataName << ":\n": std::cout << info << ":\n";
    long unsigned totSize = 1;
    std::cout << "dim: " ;
    for (auto d :dim)
    {
      totSize *= d;
      std::cout << d <<  " " ;
    }
    std::cout << ". Total " << totSize*sizeof(T)/1024.<< "kB."<< std::endl;

    printSize = printSize<totSize?printSize:totSize;

    std::cout << "First " << printSize <<" elements: " ;
    for( long long unsigned i=0; i< printSize; i++)
    {
      std::cout << vec[i] <<  " " ;
    }
    std::cout << std::endl;
    std::cout << "Last " << printSize <<" elements: " ;
    for( long long unsigned i=vec.size()-printSize; i < vec.size(); i++)
    {
      std::cout << vec[i] <<  " " ;
    }
    std::cout << std::endl;
    std::cout << "\n";
  }

  bool shrink2D()
  {
    bool shrinkable = true;
    for(long unsigned secomdDim = 0; secomdDim < dim[0]-1; secomdDim++ )
    {
       if(!std::equal (vec.data()+dim[1]*secomdDim, vec.data()+dim[1]*(secomdDim+1),
       vec.data()+dim[1]*(secomdDim+1) ))
       {
         shrinkable=false;
         return shrinkable;
       }
    }

    std::vector<T> temp(vec.data(), vec.data()+dim[1]);
    vec.swap(temp);
    dim[0]=dim[1];
    dim.resize(1);
    return true;
  }

  void swapaxes3d()
  {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (dim.size()!=3)
      throw std::runtime_error ("swapaxes3d, wrong dimension");
    std::vector<long long unsigned int> newdim(dim);
    newdim[1] = dim[0];
    newdim[0] = dim[1];


    std::vector<T> newrawdata(vec.size());
    T *pt = newrawdata.data();
    // #pragma omp parallel for: fixme with parallel a bug found
    for(long unsigned thirdDim = 0; thirdDim < newdim[0]; thirdDim++ )
    {
      for(long unsigned secomdDim = 0; secomdDim < newdim[1]; secomdDim++ )
      {
        for(long unsigned firstDim = 0; firstDim < newdim[2]; firstDim++ )
        {
          *(pt++) = vec[secomdDim*dim[1]*dim[2] + thirdDim*dim[2] + firstDim];
        }
      }
    }

    newrawdata.swap(vec);
    newdim.swap(dim);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "swapaxes3d time " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
  }
};

#ifdef __cplusplus
extern "C" {
#endif
  void* Data_new();
  void Data_delete(void* d);
  void Data_sanityCheck(void* d);
  size_t Data_getDim(void* d);
  long long unsigned* Data_getShape(void* d);
  double* Data_getVec(void* d);
#ifdef __cplusplus
}
#endif

#endif
