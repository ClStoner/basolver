#include "include/vertex.h"
#include "include/edge.h"
#include <iostream>

using namespace std;

namespace BaSolver {

unsigned long global_edge_id = 0;

Edge::Edge(int residual_dimension, int num_verticies,
           const std::vector<std::string> &verticies_types) {
    residual_dimension_ = residual_dimension;
    num_verticies_ = num_verticies;
    residual_ = new double[residual_dimension_];
    // std::cout << "residual dimension : " << residual_dimension << " , " << residual_dimension_ << " , " << ResidualDimension() <<  endl;
    if (!verticies_types.empty())
        verticies_types_ = verticies_types;
    jacobians_ = new double*[num_verticies_];
    id_ = global_edge_id++;

    Eigen::MatrixXd information(residual_dimension_, residual_dimension_);
    information.setIdentity();
    information_ = information;

    lossfunction_ = NULL;
}

Edge::~Edge() {
    // std::cout << "~Edge " <<  std::endl;
    delete[] residual_;
    for(int i = 0; i < num_verticies_; i ++)
    {
        delete[] jacobians_[i];
    }
    delete[] jacobians_;
}

double Edge::Chi2() const{
    return cost_;
}
double Edge::ComputeChi2()
{
    cost_ = 0;
    for(int i = 0; i < residual_dimension_; i ++) cost_ += (residual_[i] * residual_[i]);
    return cost_;
}

double Edge::RobustChi2() const{

    double e2 = this->Chi2();
    if(lossfunction_)
    {
        double rho[3];
        lossfunction_->Compute(e2,rho);
        e2 = rho[0];
    }
    return e2;
}
void Edge::UpdateJacRes()
{
    if(lossfunction_) 
    {
        double e2 = this->Chi2();
        double rho[3];
        lossfunction_->Compute(e2, rho);
        Corrector correct(e2, rho);
        //先修正雅克比，后修正残差
        for(int i = 0; i < num_verticies_; i ++)
        {
            if(jacobians_[i])
            {
                correct.CorrectJacobian(residual_dimension_, verticies_[i]->LocalDimension(), residual_, jacobians_[i]);
            }
        }
        correct.CorrectResiduals(residual_dimension_, residual_);
    }
}
bool Edge::CheckValid() {
    if (!verticies_types_.empty()) {
        // check type info
        for (size_t i = 0; i < verticies_.size(); ++i) {
            if (verticies_types_[i] != verticies_[i]->TypeInfo()) {
                cout << "Vertex type does not match, should be " << verticies_types_[i] <<
                     ", but set to " << verticies_[i]->TypeInfo() << endl;
                return false;
            }
        }
    }
    return true;
}

}
