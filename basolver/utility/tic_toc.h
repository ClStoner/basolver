/**
 * @file tic_toc.h
 * @author Chenglei (ClStoner@163.com)
 * @brief 统计时间消耗类
 * @version 0.1
 * @date 2022-09-06
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef BASOLVER_TIC_TOC_H
#define BASOLVER_TIC_TOC_H

#include <ctime>
#include <cstdlib>
#include <chrono>

namespace BaSolver
{

class TicToc
{
  public:
    TicToc()
    {
        tic();
    }
    void tic()
    {
        st = std::chrono::system_clock::now();
    }
    double toc()
    {
        ed = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = ed - st;
        return elapsed_seconds.count() * 1000;
    }

  private:
    std::chrono::time_point<std::chrono::system_clock> st, ed;
};

}
#endif