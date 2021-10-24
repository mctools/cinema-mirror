namespace PT=Prompt;

template <class T>
inline double PT::RandCanonical<T>::generate() const
{
  return std::generate_canonical<double,
         std::numeric_limits<double>::digits>(*(m_generator.get()));
}

template <class T>
PT::RandCanonical<T>::RandCanonical(std::shared_ptr<T> gen)
:m_generator(gen), m_seed(5489u), m_seedIsSet(false)
{
}

template <class T>
PT::RandCanonical<T>::~RandCanonical() = default;


template <class T>
inline void PT::RandCanonical<T>::setSeed(uint64_t seed)
{
  if(m_seedIsSet)
    PROMPT_THROW(BadInput, "seed is already set")
  m_seed = seed;
  m_generator.get()->seed(seed);
  m_seedIsSet=true;
}
