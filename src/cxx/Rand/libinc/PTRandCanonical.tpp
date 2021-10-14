namespace PT=Prompt;

template <class T>
inline double PT::RandCanonical<T>::generate() const
{
  return std::generate_canonical<double,
         std::numeric_limits<double>::digits>(*(m_generator.get()));
}

template <class T>
PT::RandCanonical<T>::RandCanonical(std::shared_ptr<T> gen)
:m_generator(gen)
{
}

template <class T>
PT::RandCanonical<T>::~RandCanonical() = default;
