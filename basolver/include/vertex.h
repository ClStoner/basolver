/**
 * @file vertex.h
 * @author Chenglei (ClStoner@163.com)
 * @brief 节点基类，参考了手写VIO的基类节点的实现
 * @version 0.1
 * @date 2022-09-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef BASOLVER_VERTEX_H
#define BASOLVER_VERTEX_H

#include "eigen_types.h"
#include <cstring>
namespace BaSolver{
/// @brief 全局变量，表示顶点 id 
extern unsigned long global_vertex_id;
/**
 * @brief 节点基类，对应一个parameter block，变量值以VecX存储，需要在构造时指定维度
 */
class Vertex {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * @brief 构造函数
     * @param num_dimension 顶点自身维度
     * @param local_dimension 本地参数化维度，为-1时认为与本身维度一样
     */
    explicit Vertex(int num_dimension, int local_dimension = -1);

    /**
     * @brief 析构函数
     * 
     */
    virtual ~Vertex();

    /**
     * @brief 返回变量自身维度
     * 
     * @return int 
     */
    int Dimension() const;

    /**
     * @brief 返回变量本地参数化维度
     * 
     * @return int 
     */
    int LocalDimension() const;

    /**
     * @brief 返回变量Id
     * 
     * @return unsigned long 
     */
    unsigned long Id() const { return id_; }

    /**
     * @brief 返回存储的优化变量值,返回指针，数值不可变
     * 
     * @return double const * 
     */
    double* Parameters() const { return parameters_; }

    /**
     * @brief 设置优化变量的值
     * 
     * @param params 变量值
     */
    void SetParameters(double *params) { parameters_ = params; }
    /**
     * @brief 备份当前优化变量的值
     * 
     */
    void BackUpParameters() {
        for(int i = 0; i < num_dimension_; i ++) parameters_backup_[i] = parameters_[i];
    }
    /**
     * @brief 回滚参数，用于丢弃一些迭代过程中不好的估计
     * 
     */
    void RollBackParameters() { 
        for(int i = 0; i < num_dimension_; i ++) parameters_[i] = parameters_backup_[i];
    }

    
    /**
     * @brief 重定义变量加法
     * 
     * @param delta 优化变量的变化量
     */
    virtual void Plus(const double* delta);

    /**
     * @brief 返回顶点的名称，在子类中实现
     * 
     * @return std::string 
     */
    virtual std::string TypeInfo() const = 0;
    /**
     * @brief 返回变量在排序后在BA_problem中的id
     * 
     * @return int 
     */
    int OrderingId() const { return ordering_id_; }
    /**
     * @brief 设置变量在排序后在BA_problem中的id
     * 
     * @param id 
     */
    void SetOrderingId(unsigned long id) { ordering_id_ = id; };    
    /**
     * @brief 固定该节点
     * 
     * @param fixed 
     */
    void SetFixed(bool fixed = true) {
        fixed_ = fixed;
    }
    /**
     * @brief 测试该点是否被固定
     * 
     * @return true 
     * @return false 
     */
    bool IsFixed() const { return fixed_; }
protected:
    /// @brief 实际存储的变量值
    double* parameters_;
    /// @brief 每次迭代优化中对参数进行备份，用于回滚
    double* parameters_backup_; 
    /// @brief 局部参数化维度
    int local_dimension_;
    /// 顶点自身维度 
    int num_dimension_;
    /// @brief 顶点的id，自动生成
    unsigned long id_;

    /// ordering id是在problem中排序后的id，用于寻找雅可比对应块
    /// ordering id带有维度信息，例如ordering_id=6则对应Hessian中的第6列
    /// 从零开始
    unsigned long ordering_id_ = 0;
    /// @brief 是否固定
    bool fixed_ = false;    
};


}
#endif
