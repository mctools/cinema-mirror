#include "RedisNumpy.hh"
#include <cstring>

RedisNumpy::RedisNumpy()
:m_fixed_magic( "\x93NUMPY")
{
  //fixme: make sure it is a little endian system
}

RedisNumpy::~RedisNumpy()
{

}

void RedisNumpy::makeNumpyArr_real(const uint8_t *data, unsigned len,
                  data_type dtype, const std::vector<uint64_t> &shape, std::string &npArr)
{
  npArr.reserve(len + 128); //a guess of tot size

  npArr = m_fixed_magic +'\x01'+'\x00' + "  ";
  npArr += "{'descr': '" + U32toASCII(dtype).getTypeString() + "', 'fortran_order': False, 'shape': (" ;

  for(auto v:shape)
    npArr += std::to_string(v) + ",";
  if(shape.size()>1)
    npArr.pop_back();

  npArr += "), }";

  unsigned dict_size = npArr.size();
  if(dict_size>65535)
    throw std::length_error("header in npy 1.0 format can't be larger than 65535 bytes, see https://docs.scipy.org/doc/numpy-1.14.2/neps/npy-format.html");

  //total meta info size mush be evenly divisible by 64 for alignment purposes.
  //-1, because terminated by a newline '\n'
  unsigned tot = ((dict_size/64)+1)*64;

  //sizeof("\x93NUMPY\x01\00")+sizeof(HEADER_LEN)=10 bytes
  uint16_t header_len_with_padding = tot-10;

  npArr[8] = ( unsigned char ) (header_len_with_padding & 0xff) ;
  npArr[9] = ( unsigned char ) ((header_len_with_padding>>8) & 0xff);

  unsigned padding_size = tot - dict_size - 1;
  npArr.append(padding_size,'\x20');
  npArr += "\n";

  npArr.append(reinterpret_cast<const char* >(data),len);
}
