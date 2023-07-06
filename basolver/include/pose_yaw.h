
/**
 * 
*/

#ifndef BASOLVER_POSE_YAW_H
#define BASOLVER_POSE_YAW_H

#include <memory>
#include "vertex.h"

namespace BaSolver {


class PoseYaw : public Vertex {

public : 
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    PoseYaw() : Vertex(6, 4) {}

    virtual void Plus(const double *delta) override;

    std::string TypeInfo() const {
        return "PoseYaw";
    }


};

}


#endif