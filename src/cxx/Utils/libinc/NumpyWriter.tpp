template <typename T>
void Prompt::NumpyWriter::writeNumpyFile(const std::string &filename, const std::vector<T> &data, data_type type,
                  const std::vector<uint64_t> &shape) const
{
  std::string serialised;
  makeNumpyArr(data, type, shape, serialised);
  std::ofstream outfile(filename, std::ofstream::binary);
  outfile << serialised;
  outfile.close();
}

template <typename T>
void Prompt::NumpyWriter::makeNumpyArr(const std::vector<T> &data, data_type type,
                              const std::vector<uint64_t> &shape, std::string &npArr) const
{
  makeNumpyArr_real( reinterpret_cast<const uint8_t*>(data.data()),
                data.size()*sizeof(T), type, shape , npArr );
}

template <typename T>
void Prompt::NumpyWriter::makeNumpyArr(const T *data, unsigned datasize, data_type type,
                              const std::vector<uint64_t> &shape, std::string &npArr) const
{
  makeNumpyArr_real( reinterpret_cast<const uint8_t*>(data), datasize*sizeof(T), type, shape , npArr );
}
