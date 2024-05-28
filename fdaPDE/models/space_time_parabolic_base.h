// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __SPACE_TIME_PARABOLIC_BASE_H__
#define __SPACE_TIME_PARABOLIC_BASE_H__

#include <fdaPDE/pde.h>
#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/utils.h>
using fdapde::core::pde_ptr;
using fdapde::core::Kronecker;

#include "space_time_base.h"

namespace fdapde {
namespace models {

// base class for parabolic regularization
template <typename Model>
class SpaceTimeParabolicBase : public SpaceTimeBase<Model, SpaceTimeParabolic> {
   protected:
    pde_ptr pde_ {};   // parabolic differential penalty df/dt + Lf - u
    // let m the number of time points
    DMatrix<double> s_;         // N x 1 initial condition vector
    DMatrix<double> u_;         // discretized forcing [1/DeltaT * (u_1 + R_0*s) \ldots u_n]
    SpMatrix<double> Im_;       // m x m sparse identity matrix (assembled once and cached for reuse)
    SpMatrix<double> L_;        // m x m matrix associated with the derivation in time
    double DeltaT_;             // time step (assumes equidistant points in time)
    SpMatrix<double> R0_;       // Im \kron R0 (R0: spatial mass matrix)
    SpMatrix<double> R1_;       // Im \kron R1 (R1: spatial penalty discretization)
    SpMatrix<double> R0_robin_; // Im \kron R0_robin (R0_robin: boundary mass matrix due to Robin bcs)

    SpMatrix<double> penT_;                      // discretization of the time derivative: L \kron R0
    fdapde::SparseLU<SpMatrix<double>> invR0_;   // factorization of Im \kron R0
    // discretized penalty: (Im \kron R1 + L \kron R0)^T*(I_m \kron R0)^{-1}*(Im \kron R1 + L \kron R0)
    SpMatrix<double> pen_;
   public:
    typedef SpaceTimeBase<Model, SpaceTimeParabolic> Base;
    using Base::lambda_D;   // smoothing parameter in space
    using Base::lambda_T;   // smoothing parameter in time
    using Base::model;      // underlying model object
    using Base::time_;      // time interval [0,T]
    using Base::df_;        // model's data
    // constructor
    SpaceTimeParabolicBase() = default;
    SpaceTimeParabolicBase(const pde_ptr& parabolic_penalty) :
      pde_(parabolic_penalty), Base(parabolic_penalty.time_domain()) { }
    // init data structure related to parabolic regularization
    void init_regularization() {
        pde_.init();
        s_ = pde_.initial_condition();   // derive initial condition from parabolic problem
        std::size_t m_ = time_.rows();   // number of time points
        DeltaT_ = time_[1] - time_[0];   // time step (assuming equidistant points)

        // assemble once the m x m identity matrix and cache for fast access
        Im_.resize(m_, m_);
        Im_.setIdentity();
        // assemble matrix associated with derivation in time L_
        // [L_]_{ii} = 1/DeltaT for i \in {1 ... m} and [L_]_{i,i-1} = -1/DeltaT for i \in {1 ... m-1}
        std::vector<fdapde::Triplet<double>> triplet_list;
        triplet_list.reserve(2 * m_);
        // start assembly loop
        double invDeltaT = 1.0 / DeltaT_;
        triplet_list.emplace_back(0, 0, invDeltaT);
        for (std::size_t i = 1; i < m_; ++i) {
            triplet_list.emplace_back(i, i, invDeltaT);
            triplet_list.emplace_back(i, i - 1, -invDeltaT);
        }
        L_.resize(m_, m_);
        L_.setFromTriplets(triplet_list.begin(), triplet_list.end());
        L_.makeCompressed();
        // compute tensorized matrices
        R0_ = Kronecker(Im_, pde_.mass());
        R1_ = Kronecker(Im_, pde_.stiff());
        R0_robin_ = Kronecker(Im_, pde_.mass_robin());
        // correct first n rows of discretized force as (u_1 + R0*s/DeltaT)
        u_ = pde_.force();
        u_.block(0, 0, model().n_basis(), 1) += (1.0 / DeltaT_) * (pde_.mass() * s_);
    }
    // setters
    void set_penalty(const pde_ptr& pde) {
        pde_ = pde;
        model().runtime().set(runtime_status::require_penalty_init);
    }
    void set_dirichlet_bc(SparseBlockMatrix<double, 2, 2>& A, DVector<double>& b);
    void set_dirichlet_bc(DVector<double>& b);
    void set_dirichlet_bc(SparseBlockMatrix<double, 2, 2>& A, DVector<double>& b, int k);
    void set_dirichlet_bc(DVector<double>& b, int k);
    // getters
    const pde_ptr& pde() const { return pde_; }   // regularizing term df/dt + Lf - u
    const SpMatrix<double>& R0() const { return R0_; }
    const SpMatrix<double>& R1() const { return R1_; }
    const SpMatrix<double> R1_step(const DVector<double>& f) const { return pde_.stiff_step(f); };   // stiff matrix evaluated in f (usuful when we have a nonlinearity)
    const SpMatrix<double> R1_step_kronecker(const DVector<double>& f) { // set R1_ to the stiff matrix evaluated in f and returns it (usuful when we have a nonlinearity)
        std::size_t m = time_.rows();  // number of time points
        std::size_t n = n_basis();     // number of dofs
        SpMatrix<double> I(m, m);
        R1_ = SpMatrix<double>(n*m, n*m);
        for (std::size_t i=0; i<m; ++i){
            I.coeffRef(i,i) = 1;
            R1_ += Kronecker(I, pde_.stiff_step(f.block(n*i, 0, n, 1)));
            I.coeffRef(i,i) = 0;
        }
            
        return R1_;
    };
    const DMatrix<double>& dirichlet_boundary_data() const { return pde_.dirichlet_boundary_data(); };
    std::size_t n_basis() const { return pde_.n_dofs(); }   // number of basis functions
    std::size_t n_spatial_basis() const { return pde_.n_dofs(); }
    const SpMatrix<double>& L() const { return L_; }
    const DMatrix<double>& u() const { return u_; }   // discretized force corrected by initial conditions
    const DMatrix<double>& u_neumann() const { return pde_.force_neumann(); }  // discretized neumann
    const DMatrix<double>& u_robin() const { return pde_.force_robin(); }      // discretized robin force
    const SpMatrix<double>& R0_robin() const { return R0_robin_; }
    const DMatrix<double>& s() { return s_; }   // initial condition
    double DeltaT() const { return DeltaT_; }

    // computes and cache matrices (Im \kron R0)^{-1} and L \kron R0, returns the discretized penalty P =
    // \lambda_D*((Im \kron R1 + \lambda_T*(L \kron R0))^T*(I_m \kron R0)^{-1}*(Im \kron R1 + \lambda_T*(L \kron R0)))
    SpMatrix<double> P() {
        if (is_empty(pen_)) {   // compute once and cache result
            invR0_.compute(R0());
            penT_ = Kronecker(L_, pde_.mass());
        }
        return lambda_D() * (R1() + lambda_T() * penT_).transpose() * invR0_.solve(R1() + lambda_T() * penT_);
    }
  
    // destructor
    virtual ~SpaceTimeParabolicBase() = default;
};

// impose boundary condition to the whole matrix and vector (monolithic approach case)
template <typename Model> void SpaceTimeParabolicBase<Model>::set_dirichlet_bc(SparseBlockMatrix<double, 2, 2>& A, DVector<double>& b) {
    // do the is_init thing

    std::size_t n_block = A.rows() / 2;
    std::size_t m = time_.rows();
    auto matrix = pde_.matrix_bc_Dirichlet();
    std::size_t n = matrix.rows();

    // std::cout << "n*m = n_block = " << n_block << std::endl;
    // std::cout << "timesteps = m = " << m << std::endl;
    // std::cout << "#dofs = n = " << n << std::endl;

    /* auto A_old = A;
    auto b_old = b; */

    if (!is_empty(pde_.dirichlet_boundary_data())) {
        for (std::size_t i = 0; i < n; ++i) {
            if (matrix(i,0) == 1) {
                for (std::size_t j = 0; j < m; ++j){
                    A.block(0,0).row(i + j*n) *= 0;  // zero all entries of this row
                    A.block(0,1).row(i + j*n) *= 0;
                    A.coeffRef(i + j*n, i + j*n) = 1;   // set diagonal element to 1 to impose equation u_j = b_j

                    A.block(1,0).row(i + j*n) *= 0;  // zero all entries of this row
                    A.block(1,1).row(i + j*n) *= 0;
                    A.coeffRef(i + n_block + j*n, i + n_block + j*n) = 1;

                    // dirichlet_boundary_data is a matrix D s.t. D_{i,j} = dirichlet datum at node i, timstep j
                    b.coeffRef(i + j*n) = pde_.dirichlet_boundary_data()(i + j*n, 0);   // impose boundary value
                    b.coeffRef(i + n_block + j*n) = 0;
                }
            }
        }
    }

    // check the correctness of the method
    /* for(size_t i=0; i<n_block; ++i){
        for(size_t j=0; j<n_block; ++j){
            // block 1
            if (matrix(i,0) == 1) { // if "i" is a Dirichlet boundary node
                if (i != j) { // not on the diagonal
                    if (A.coeffRef(i,j) != 0) std::cout << "A(" << i << ", " << j << ") != 0" << std::endl;
                    if (A.coeffRef(i+n_block,j) != 0) std::cout << "A(" << i+n_block << ", " << j << ") != 0" << std::endl;
                    if (A.coeffRef(i,j+n_block) != 0) std::cout << "A(" << i << ", " << j+n_block << ") != 0" << std::endl;
                    if (A.coeffRef(i+n_block,j+n_block) != 0) std::cout << "A(" << i+n_block << ", " << j+n_block << ") != 0" << std::endl;

                }
                else {
                    if (A.coeffRef(i,j) != 1) std::cout << "A(" << i << ", " << j << ") != 1" << std::endl;
                    if (A.coeffRef(i+n_block,j+n_block) != 1) std::cout << "A(" << i << ", " << j << ") != 1" << std::endl;
                    if (A.coeffRef(i+n_block,j) != 0) std::cout << "A(" << i+n_block << ", " << j << ") != 0" << std::endl;
                    if (A.coeffRef(i,j+n_block) != 0) std::cout << "A(" << i << ", " << j+n_block << ") != 0" << std::endl;
                }
            }
            else {
                if (A_old.coeffRef(i,j) != A.coeffRef(i,j)) std::cout << "PROBLEM!" << std::endl;
                if (A_old.coeffRef(i+n_block,j) != A.coeffRef(i+n_block,j)) std::cout << "PROBLEM!" << std::endl;
                if (A_old.coeffRef(i,j+n_block) != A.coeffRef(i,j+n_block)) std::cout << "PROBLEM!" << std::endl;
                if (A_old.coeffRef(i+n_block,j+n_block) != A.coeffRef(i+n_block,j+n_block)) std::cout << "PROBLEM!" << std::endl;
            }
        }
    } */

    return;
}

// impose boundary condition to the whole vector (monolithic approach case)
template <typename Model> void SpaceTimeParabolicBase<Model>::set_dirichlet_bc(DVector<double>& b) {
    // do the is_init thing

    std::size_t n_block = b.rows() / 2;
    std::size_t m = time_.rows();
    auto matrix = pde_.matrix_bc_Dirichlet();
    std::size_t n = matrix.rows();

    if (!is_empty(pde_.dirichlet_boundary_data())) {
        for (std::size_t i = 0; i < n; ++i) {
            if (matrix(i,0) == 1) {
                for (std::size_t j = 0; j < m; ++j){
                    // dirichlet_boundary_data is a matrix D s.t. D_{i,j} = dirichlet datum at node i, timstep j
                    b.coeffRef(i + j*n) = pde_.dirichlet_boundary_data()(i + j*n, 0);   // impose boundary value
                    b.coeffRef(i + n_block + j*n) = 0;
                }
            }
        }
    }

    return;
}

// set dirichlet boundary conditions when solving the problem concerning timestep k
template <typename Model> void SpaceTimeParabolicBase<Model>::set_dirichlet_bc(SparseBlockMatrix<double, 2, 2>& A, DVector<double>& b, int k) {
    // do the is_init thing

    std::size_t n = A.rows() / 2;
    auto matrix = this->pde().matrix_bc_Dirichlet();

    if (!is_empty(this->pde().dirichlet_boundary_data())) {
        for (std::size_t i = 0; i < n; ++i) {
            if (matrix(i,0) == 1) {
                A.block(0,0).row(i) *= 0;  // zero all entries of this row
                A.block(0,1).row(i) *= 0;
                A.coeffRef(i, i) = 1;   // set diagonal element to 1 to impose equation u_j = b_j

                A.block(1,0).row(i) *= 0;  // zero all entries of this row
                A.block(1,1).row(i) *= 0;
                A.coeffRef(i + n, i + n) = 1;

                // dirichlet_boundary_data is a matrix D s.t. D_{i,j} = dirichlet datum at node i, timstep j
                b.coeffRef(i) = this->pde().dirichlet_boundary_data()(i + k*n, 0);   // impose boundary value
                b.coeffRef(i + n) = 0;
            }
        }
    }

    return;
}

// set dirichlet boundary conditions to vector b when solving the problem concerning timestep k
template <typename Model> void SpaceTimeParabolicBase<Model>::set_dirichlet_bc(DVector<double>& b, int k) {
    // do the is_init thing

    std::size_t n = b.rows() / 2;
    auto matrix = this->pde().matrix_bc_Dirichlet();

    if (!is_empty(this->pde().dirichlet_boundary_data())) {
        for (std::size_t i = 0; i < n; ++i) {
            if (matrix(i,0) == 1) {
                // dirichlet_boundary_data is a matrix D s.t. D_{i,j} = dirichlet datum at node i, timstep j
                b.coeffRef(i) = this->pde().dirichlet_boundary_data()(i + k*n, 0);   // impose boundary value
                b.coeffRef(i + n) = 0;
            }
        }
    }

    return;
}
  
}   // namespace models
}   // namespace fdapde

#endif   // __SPACE_TIME_PARABOLIC_BASE_H__
