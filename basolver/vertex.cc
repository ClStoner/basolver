#include "include/vertex.h"
#include <iostream>

namespace BaSolver {

unsigned long global_vertex_id = 0;

Vertex::Vertex(int num_dimension, int local_dimension) {
    num_dimension_ = num_dimension;
    local_dimension_ = local_dimension > 0 ? local_dimension : num_dimension;
    // parameters_ = new double[num_dimension_];
    parameters_backup_ = new double[num_dimension_];
    id_ = global_vertex_id++;

//    std::cout << "Vertex construct num_dimension: " << num_dimension
//              << " local_dimension: " << local_dimension << " id_: " << id_ << std::endl;
}

Vertex::~Vertex() {
    // delete[] parameters_;
    // std::cout << "~Vertex" << std::endl;
    delete[] parameters_backup_;
    // std::cout << "succeed Vertex" << std::endl;
}

int Vertex::Dimension() const {
    return num_dimension_;
}

int Vertex::LocalDimension() const {
    return local_dimension_;
}

void Vertex::Plus(const double *delta) {
    for(int i = 0; i < num_dimension_; i ++) parameters_[i] += delta[i];
}

}