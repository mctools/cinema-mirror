#include "PTException.hh"

namespace Prompt {

  namespace Error {

    Exception::Exception(const std::string& msg, const char * f, unsigned l) throw()
      : std::runtime_error(msg),
        m_file(f),
        m_lineno(l)
    {
    }

    Exception::Exception(const char * msg,  const char * f, unsigned l) throw()
      : std::runtime_error(msg),
        m_file(f),
        m_lineno(l)
    {
    }

    Exception::Exception( const Exception & o ) throw()
      : std::runtime_error(o),
        m_file(o.m_file),
        m_lineno(o.m_lineno)
    {
    }

    Exception & Exception::operator= ( const Exception & o ) throw()
    {
      std::runtime_error::operator=(o);
      m_file = o.m_file;
      m_lineno = o.m_lineno;
      return *this;
    }

    Exception::~Exception() throw()
    {
    }

    FileNotFound::~FileNotFound() throw() {}
    MissingInfo::~MissingInfo() throw() {}
    DataLoadError::~DataLoadError() throw() {}
    LogicError::~LogicError() throw() {}
    CalcError::~CalcError() throw() {}
    BadInput::~BadInput() throw() {}
  }

}
