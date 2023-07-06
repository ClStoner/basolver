#pragma once

#include <ctime>
#include <cstdlib>
#include <chrono>

namespace utility
{
class TicToc
{
  public:
    TicToc()
    {
        tic();
    }

    inline void tic()
    {
        start = std::chrono::system_clock::now();
    }

    inline double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

  private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};
}  // namespace utility