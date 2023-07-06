/**
 * @file edge.h
 * @author Chenglei (ClStoner@163.com)
 * @brief 约束基类，参考手写VIO中基类约束的实现
 * @version 0.1
 * @date 2022-09-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef BASOLVER_EDGE_H
#define BASOLVER_EDGE_H

#include <memory>
#include <string>
#include "eigen_types.h"
#include <eigen3/Eigen/Dense>
#include "loss_function.h"
#include "corrector.h"
#include "vertex.h"
#include <iostream>
#include "mymatrix.h"
#include "parameters.h"
namespace BaSolver {

/**
 * @brief 全局变量，表示约束ID
 * 
 */
extern unsigned long global_edge_id;

class Vertex;

/**
 * @brief 各种约束的基类
 * 
 */
class Edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * @brief 构造函数
     * 
     * @param residual_dimension 残差维度
     * @param num_verticies 节点数量
     * @param verticies_types 节点类型
     */
    explicit Edge(int residual_dimension, int num_verticies,
                  const std::vector<std::string> &verticies_types = std::vector<std::string>());

    /**
     * @brief 析构函数
     * 
     */
    virtual ~Edge();

    /**
     * @brief 返回约束id
     * 
     * @return unsigned long 
     */
    unsigned long Id() const { return id_; }

    /**
     * 设置一个顶点
     * @param vertex 对应的vertex对象
     */
    
    /**
     * @brief 添加一个节点
     * 
     * @param vertex 节点
     * @return true 
     * @return false 
     */
    bool AddVertex(std::shared_ptr<Vertex> vertex) {
        verticies_.emplace_back(vertex);
        return true;
    }

    /**
     * @brief 设置一些节点
     * 
     * @param vertices 节点，按引用顺序排列
     * @return true 
     * @return false 
     */
    bool SetVertex(const std::vector<std::shared_ptr<Vertex>> &vertices) {
        verticies_ = vertices;
        for(int i = 0; i < num_verticies_; i ++)
        {   
            int len = residual_dimension_ * vertices[i]->LocalDimension();
            jacobians_[i] = new double[len];
        }
        return true;
    }

    /**
     * @brief 返回第i个节点
     * 
     * @param i 节点id
     * @return std::shared_ptr<Vertex> 
     */
    std::shared_ptr<Vertex> GetVertex(int i) {
        return verticies_[i];
    }

    /**
     * @brief 返回所有顶点
     * 
     * @return std::vector<std::shared_ptr<Vertex>> 
     */
    std::vector<std::shared_ptr<Vertex>> Verticies() const {
        return verticies_;
    }
    /**
     * @brief 返回关联顶点个数
     * 
     * @return size_t 
     */
    size_t NumVertices() const { return verticies_.size(); }

    /**
     * @brief 返回约束的类型信息，在子类中实现
     * 
     * @return std::string 
     */
    virtual std::string TypeInfo() const = 0;

    /**
     * @brief 计算残差，由子类实现
     * 
     */
    virtual void ComputeResidual() = 0;


    /**
     * @brief 计算残差和雅克比雅可比、完成残差雅克比核函数矫正以及填充信息矩阵。 由子类实现，
     * 不支持自动求导，需要实现每个子类的雅可比计算方法
     * 
     */
    virtual void ComputeJacobians() = 0;
    /**
     * @brief 边缘化时计算残差以及雅克比、完成残差雅克比核函数矫正
     * 
     */
    virtual void ComputeOnlyJacobians() = 0;


    /**
     * @brief 利用残差计算平方误差，会乘以信息矩阵
     * 
     * @return double 
     */
    double ComputeChi2();
    /**
     * @brief 返回平方误差
     * 
     * @return double 
     */
    double Chi2() const;
    /**
     * @brief 返回使用损失核函数后的平方误差
     * 
     * @return double 
     */
    double RobustChi2() const;
    /**
     * @brief 利用损失核函数矫正雅克比和残差
     * 
     */
    void UpdateJacRes();

    /**
     * @brief 返回残差
     * 
     * @return double* 
     */
    double* Residual() const { 
        return residual_; 
    }
    int ResidualDimension() const {
        return residual_dimension_;
    }
    /**
     * @brief 返回雅可比
     * 
     * @return double**
     */
    double** Jacobians() const { return jacobians_; }

    /**
     * @brief 设置信息矩阵
     * 
     * @param information 
     */
    void SetInformation(const MatXX &information) {
        information_ = information;
    }

    /**
     * @brief 返回信息矩阵
     * 
     * @return MatXX 
     */
    MatXX Information() const {
        return information_;
    }
    /**
     * @brief 设置约束的损失和函数
     * 
     * @param ptr 
     */
    void SetLossFunction(LossFunction* ptr){ lossfunction_ = ptr; }
    /**
     * @brief 返回约束的损失核函数
     * 
     * @return LossFunction* 
     */
    LossFunction* GetLossFunction(){ return lossfunction_;}


    /**
     * @brief 检查约束信息是否全部设置
     * 
     * @return true 
     * @return false 
     */
    bool CheckValid();

   
protected:
    /// @brief edge id
    unsigned long id_;  
    /// @brief 各节点类型信息，用于debug
    std::vector<std::string> verticies_types_; 
    /// @brief 此约束对应的节点
    std::vector<std::shared_ptr<Vertex>> verticies_; 
    /// @brief 残差
    double *residual_; 
    /// @brief 雅可比
    double **jacobians_; 
    /// @brief 残差维度
    int residual_dimension_;
    /// @brief 优化变量的个数
    int num_verticies_;
    /// @brief 信息矩阵
    MatXX information_;            
    /// @brief 损失核函数
    LossFunction *lossfunction_;
    /// @brief 平方误差
    double cost_;
};

}

#endif
