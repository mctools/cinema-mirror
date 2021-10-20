#ifndef Prompt_Singleton_hh
#define Prompt_Singleton_hh

namespace Prompt {

template<typename T>
class Singleton
{
public:
    static T& getInstance()
    {
        static T value;
        return value;
    }

private:
    Singleton();
    ~Singleton();
};

}

#endif
