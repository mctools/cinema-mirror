#ifndef Prompt_ProgressMonitor_hh
#define Prompt_ProgressMonitor_hh

#include <string>
#include <chrono>

namespace Prompt {
  class ProgressMonitor {
  public:
    ProgressMonitor(const std::string& name, double numTask, double m_interval=0.1);
    ~ProgressMonitor();
    void OneTaskCompleted();
    double getEstimated_ms() {return m_estimated_ms;}
  private:
    std::string m_name;
    double m_numTask, m_currentTask, m_completedRatio, m_interval, m_estimated_ms;
    std::chrono::steady_clock::time_point m_begin;
  };
}
#endif
