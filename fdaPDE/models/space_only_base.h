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

#ifndef __SPACE_ONLY_BASE_H__
#define __SPACE_ONLY_BASE_H__

#include <fdaPDE/utils.h>
#include <fdaPDE/pde.h>
#include <fdaPDE/linear_algebra.h>
#include "model_base.h"
using fdapde::core::pde_ptr;
using fdapde::core::lump;
using fdapde::core::SparseBlockMatrix;

namespace fdapde {
namespace models {

// abstract base interface for any *space-only* fdaPDE statistical model.
template <typename Model> class SpaceOnlyBase : public ModelBase<Model> {
   public:
    using Base = ModelBase<Model>;
    static constexpr int n_lambda = n_smoothing_parameters<SpaceOnly>::value;
    using Base::lambda;       // dynamic sized smoothing parameter vector
    using Base::model;        // underlying model object
    using Base::set_lambda;   // dynamic sized setter for \lambda
    // constructor
    SpaceOnlyBase() = default;
    SpaceOnlyBase(const pde_ptr& space_penalty) : pde_(space_penalty) {};
    void init_regularization() {
        pde_.init();
        if (mass_lumping) { R0_lumped_ = lump(pde_.mass()); }   // lump mass matrix if requested
    }
    // public flags
    bool mass_lumping = false;
    // setters
    void set_lambda(const SVector<n_lambda>& lambda) {
        if(lambda_ == lambda) return;
        model().runtime().set(runtime_status::is_lambda_changed);
        lambda_ = lambda;
    }
    void set_lambda_D(double lambda_D) { set_lambda(SVector<n_lambda>(lambda_D)); }
    void set_penalty(const pde_ptr& pde) {
        pde_ = pde;
        model().runtime().set(runtime_status::require_penalty_init);
    }
    void set_dirichlet_bc(SparseBlockMatrix<double, 2, 2>& A, DVector<double>& b);
    // getters
    SVector<n_lambda> lambda() const { return lambda_; }
    double lambda_D() const { return lambda_[0]; }
    const SpMatrix<double>& R0() const { return mass_lumping ? R0_lumped_ : pde_.mass(); }   // mass matrix
    const SpMatrix<double>& R1() const { return pde_.stiff(); }                // discretization of differential operator L
    const DMatrix<double>& u() const { return pde_.force(); }                  // discretization of forcing term u
    const DMatrix<double>& u_neumann() const { return pde_.force_neumann(); }  // discretization of forcing term u
    const DMatrix<double>& u_robin() const { return pde_.force_robin(); }      // discretization of forcing term u
    const SpMatrix<double>& R0_robin() const { return pde_.mass_robin(); }     // buondary mass matrix due to Robin bcs
    inline std::size_t n_temporal_locs() const { return 1; }                   // number of time instants
    std::size_t n_basis() const { return pde_.n_dofs(); };                     // number of basis functions
    std::size_t n_spatial_basis() const { return n_basis(); }                  // number of basis functions in space
    const pde_ptr& pde() const { return pde_; }                                // regularizing term Lf - u
    const fdapde::SparseLU<SpMatrix<double>>& invR0() const {                  // LU factorization of mass matrix R0
        if (!invR0_) { invR0_.compute(R0()); }
        return invR0_;
    }
    const SpMatrix<double>& PD() const {   // space-penalty component (R1^T*R0^{-1}*R1)
        if (is_empty(P_)) { P_ = R1().transpose() * invR0().solve(R1()); }
        return P_;
    }
    // evaluation of penalty term \lambda*(R1^\top*R0^{-1}*R1) at \lambda
    auto P(const SVector<n_lambda>& lambda) const { return lambda[0] * PD(); }
    auto P() const { return P(lambda()); }
    const DMatrix<int>& matrix_bc_Dirichlet() const { return matrix_bc_Dirichlet_; }

    // destructor
    virtual ~SpaceOnlyBase() = default;
   protected:
    pde_ptr pde_ {};               // differential penalty in space Lf - u
    mutable SpMatrix<double> P_;   // discretization of penalty term: R1^T*R0^{-1}*R1
    mutable fdapde::SparseLU<SpMatrix<double>> invR0_;
    SpMatrix<double> R0_lumped_;   // lumped mass matrix, if mass_lumping == true, empty otherwise
    SVector<n_lambda> lambda_ = SVector<n_lambda>::Zero();
    DMatrix<int> matrix_bc_Dirichlet_; // matrix that has 1 if a dof is Dirichlet and 0 otherwise
};

template <typename Model> void SpaceOnlyBase <Model>::set_dirichlet_bc(SparseBlockMatrix<double, 2, 2>& A, DVector<double>& b) {
    // do the is_init thing

    std::size_t n = A.rows() / 2;
    auto matrix = pde_.matrix_bc_Dirichlet();

    /* auto A_old = A;
    auto b_old = b; */
    if (!is_empty(pde_.dirichlet_boundary_data())) {
        for (std::size_t i = 0; i < n; ++i) {
            if (matrix(i,0) == 1) {
                A.block(0,0).row(i) *= 0;  // zero all entries of this row
                A.block(0,1).row(i) *= 0;
                A.coeffRef(i, i) = 1;   // set diagonal element to 1 to impose equation u_j = b_j

                A.block(1,0).row(i) *= 0;  // zero all entries of this row
                A.block(1,1).row(i) *= 0;
                A.coeffRef(i + n, i + n) = 1;

                // dirichlet_boundary_data is a matrix D s.t. D_{i,j} = dirichlet datum at node i, timstep j
                b.coeffRef(i) = pde_.dirichlet_boundary_data()(i, 0);   // impose boundary value
                b.coeffRef(i + n) = 0;
            }
        }
    }

    // check the correctness of the method
    /* for(size_t i=0; i<n; ++i){
        for(size_t j=0; j<n; ++j){
            // block 1
            if (matrix(i,0) == 1) { // if "i" is a Dirichlet boundary node
                if (i != j) { // not on the diagonal
                    if (A.coeffRef(i,j) != 0) std::cout << "A(" << i << ", " << j << ") != 0" << std::endl;
                    if (A.coeffRef(i+n,j) != 0) std::cout << "A(" << i+n << ", " << j << ") != 0" << std::endl;
                    if (A.coeffRef(i,j+n) != 0) std::cout << "A(" << i << ", " << j+n << ") != 0" << std::endl;
                    if (A.coeffRef(i+n,j+n) != 0) std::cout << "A(" << i+n << ", " << j+n << ") != 0" << std::endl;

                }
                else {
                    if (A.coeffRef(i,j) != 1) std::cout << "A(" << i << ", " << j << ") != 1" << std::endl;
                    if (A.coeffRef(i+n,j+n) != 1) std::cout << "A(" << i << ", " << j << ") != 1" << std::endl;
                    if (A.coeffRef(i+n,j) != 0) std::cout << "A(" << i+n << ", " << j << ") != 0" << std::endl;
                    if (A.coeffRef(i,j+n) != 0) std::cout << "A(" << i << ", " << j+n << ") != 0" << std::endl;
                }
            }
            else {
                if (A_old.coeffRef(i,j) != A.coeffRef(i,j)) std::cout << "PROBLEM!" << std::endl;
                if (A_old.coeffRef(i+n,j) != A.coeffRef(i+n,j)) std::cout << "PROBLEM!" << std::endl;
                if (A_old.coeffRef(i,j+n) != A.coeffRef(i,j+n)) std::cout << "PROBLEM!" << std::endl;
                if (A_old.coeffRef(i+n,j+n) != A.coeffRef(i+n,j+n)) std::cout << "PROBLEM!" << std::endl;
            }
        }
    } */

    return;
}

}   // namespace models
}   // namespace fdapde

#endif   // __SPACE_ONLY_BASE_H__
