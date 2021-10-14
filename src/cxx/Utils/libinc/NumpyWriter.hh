#ifndef NumpyWriter_hh
#define NumpyWriter_hh

#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <fstream>

namespace Prompt {

  //!The NumpyWriter class is able to pack C type arrays into the native numpy npy v1.0 format. By xx cai
  //!It can be used to transceive numpy array over Redis store between C++ and python without any encoding/decoding

  //!defintion of the format can be found at
  //!https://docs.scipy.org/doc/numpy-1.14.2/neps/npy-format.html
  //!the implementation in numpy is at
  //!https://github.com/numpy/numpy/blob/master/numpy/lib/format.py

  //!Limitations
  //!1. limited by format, one chunk of the converted data can only represent one numpy array.
  //!2. limited by implementation, all numpy array element must have the same type and not compressed.
  //!3. limited by implementation, only creating not parsing numpy data.
  //!4. limited by implementation, must by used in little endian system.

  class NumpyWriter {
    //'<'' means little endian
    //0x3c63 ascii for <c
    //0x3c66 ascii for <f
    //0x3c69 ascii for <i
    //0x3c75 ascii for <u

    //0x31 ascii for 1
    //0x32 ascii for 2
    //0x34 ascii for 4
    //0x38 ascii for 8

    //0x3c633136 for <c16

  public:
    enum data_type : uint32_t {
      c8 = 0x38633c,
      f4 = 0x34663c,
      f8 = 0x38663c,
      i1 = 0x31693c,
      i2 = 0x32693c,
      i4 = 0x34693c,
      i8 = 0x38693c,
      u1 = 0x31753c,
      u2 = 0x32753c,
      u4 = 0x34753c,
      u8 = 0x38753c,
      c16 = 0x3631633c
   };
  public:
    NumpyWriter();
    virtual ~NumpyWriter();

    template <typename T>
    void makeNumpyArr(const std::vector<T> &data, data_type type,
                      const std::vector<uint64_t> &shape, std::string &npArr) const;

    template <typename T>
    void makeNumpyArr(const T *data, unsigned datasize, data_type type,
                      const std::vector<uint64_t> &shape, std::string &npArr) const;

    template <typename T>
    void writeNumpyFile(const std::string &filename, const std::vector<T> &data, data_type type,
                      const std::vector<uint64_t> &shape) const;

  private:
    const std::string m_fixed_magic;
    void makeNumpyArr_real(const uint8_t  *data, unsigned len,
                      data_type type, const std::vector<uint64_t> &shape, std::string &npArr) const;
  private:
    //use the data_type to construct. getTypeString retures the ascii name of the type in npy format
    union U32toASCII {
      uint32_t u32;
      char  ascii[4];
      U32toASCII(uint32_t a) : u32(a) {};
      std::string getTypeString() {
        std::string type_str;
        for(unsigned i=0;i<4;i++) {
          if(!ascii[i])
            return type_str;
          type_str += *(ascii+i);
        }
        if(type_str.empty()) //shouldn't really reach here
          throw std::invalid_argument("data_type ascii array can't be all zero");
        return type_str;
      }
    };
  };
}

#include "NumpyWriter.tpp"

#endif
