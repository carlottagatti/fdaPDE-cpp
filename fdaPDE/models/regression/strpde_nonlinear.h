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

#ifndef __STRPDE_NONLINEAR_H__
#define __STRPDE_NONLINEAR_H__

#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/pde.h>
#include <fdaPDE/utils.h>

#include <memory>
#include <type_traits>

#include "../model_base.h"
#include "../model_macros.h"
#include "../model_traits.h"
#include "../sampling_design.h"
#include "regression_base.h"
using fdapde::core::BlockVector;
using fdapde::core::Kronecker;
using fdapde::core::SMW;
using fdapde::core::SparseBlockMatrix;
using fdapde::core::KroneckerTensorProduct;
using fdapde::core::SplineBasis;

namespace fdapde {
namespace models {

// STRPDE_NonLinear model signature
template <typename RegularizationType, typename SolutionPolicy> class STRPDE_NonLinear;

// implementation of STRPDE for parabolic nonlinear space-time regularization, monolithic approach
template <>
class STRPDE_NonLinear<SpaceTimeParabolic, monolithic> :
    public RegressionBase<STRPDE_NonLinear<SpaceTimeParabolic, monolithic>, SpaceTimeParabolic> {
   private:
    SparseBlockMatrix<double, 2, 2> A_ {};      // system matrix of non-parametric problem (2N x 2N matrix)
    fdapde::SparseLU<SpMatrix<double>> invA_;   // factorization of matrix A
    DVector<double> b_ {};                      // right hand side of problem's linear system (1 x 2N vector)
    SpMatrix<double> L_;                        // L \kron R0
   public:
    using RegularizationType = SpaceTimeParabolic;
    using Base = RegressionBase<STRPDE_NonLinear<RegularizationType, monolithic>, RegularizationType>;
    // import commonly defined symbols from base
    IMPORT_REGRESSION_SYMBOLS;
    using Base::L;          // [L]_{ii} = 1/DeltaT for i \in {1 ... m} and [L]_{i,i-1} = -1/DeltaT for i \in {1 ... m-1}
    using Base::lambda_D;   // smoothing parameter in space
    using Base::lambda_T;   // smoothing parameter in time
    using Base::n_temporal_locs;   // number of time instants m defined over [0,T]
    using Base::s;                 // initial condition
    // constructor
    STRPDE_NonLinear() = default;
    STRPDE_NonLinear(const pde_ptr& pde, Sampling s) : Base(pde, s) {};

    // internal utilities
    DMatrix<double> y(std::size_t k) const { return y().block(n_spatial_locs() * k, 0, n_spatial_locs(), 1); }
    DMatrix<double> u(std::size_t k) const { return u_.block(n_basis() * k, 0, n_basis(), 1); }
    DMatrix<double> u_neumann(std::size_t k) const { return pde().force_neumann().block(n_basis() * k, 0, n_basis(), 1); }
    DMatrix<double> u_robin(std::size_t k) const { return pde().force_robin().block(n_basis() * k, 0, n_basis(), 1); }
    const SpMatrix<double>& R0() const { return pde_.mass(); }    // mass matrix in space
    const SpMatrix<double>& R1() const { return pde_.stiff(); }   // discretization of differential operator L

    void init_model() {   // update model object in case of **structural** changes in its definition
        // assemble system matrix for the nonparameteric part of the model
        if (is_empty(L_)) L_ = Kronecker(L(), pde().mass());
        A_ = SparseBlockMatrix<double, 2, 2>(
          -PsiTD() * W() * Psi(), lambda_D() * (R1_step(s_) + R0_robin() + lambda_T() * L_).transpose(),
          lambda_D() * (R1_step(s_) + R0_robin() + lambda_T() * L_), lambda_D() * R0());
        // cache system matrix for reuse
        invA_.compute(A_);
        // prepare rhs of linear system
        b_.resize(A_.rows());
        b_.block(A_.rows() / 2, 0, A_.rows() / 2, 1) = lambda_D() * (u(0) + u_neumann(0) + u_robin(0));
        return;
    }
    void update_to_weights() {   // update model object in case of changes in the weights matrix
        // adjust north-west block of matrix A_ and factorize
        A_.block(0, 0) = -PsiTD() * W() * Psi();
        invA_.compute(A_);
        return;
    }
    void update_model(size_t k) {   // update model object in case of **structural** changes in its definition
        // assemble system matrix for the nonparameteric part of the model
        A_ = SparseBlockMatrix<double, 2, 2>(
          -PsiTD() * W() * Psi(), lambda_D() * (R1_step(f_.block(n_basis()*(k-1), 0, n_basis(), 1)) + R0_robin() + lambda_T() * L_).transpose(),
          lambda_D() * (R1_step(f_.block(n_basis()*(k-1), 0, n_basis(), 1)) + R0_robin() + lambda_T() * L_), lambda_D() * R0());
        // cache system matrix for reuse
        invA_.compute(A_);
        // prepare rhs of linear system
        b_.resize(A_.rows());
        b_.block(A_.rows() / 2, 0, A_.rows() / 2, 1) = lambda_D() * (u(k) + u_neumann(k) + u_robin(k));
        return;
    }
    void solve() {
        fdapde_assert(y().rows() != 0);
        DVector<double> sol;             // room for problem' solution
        
        // loop over time
        for (size_t k=0; k<time_.rows(); k++) {
            if (k > 0) update_model(k);

            if (!Base::has_covariates()) {   // nonparametric case
                // update rhs of STR-PDE linear system
                b_.block(0, 0, A_.rows() / 2, 1) = -PsiTD() * W() * y();
                // solve linear system A_*x = b_
                sol = invA_.solve(b_);
                f_.block(n_basis()*k, 0, n_basis(), 1) = sol.head(A_.rows() / 2);
            } else {   // parametric case
                // rhs of STR-PDE linear system
                b_.block(0, 0, A_.rows() / 2, 1) = -PsiTD() * lmbQ(y());   // -\Psi^T*D*Q*z
                // matrices U and V for application of woodbury formula
                U_ = DMatrix<double>::Zero(A_.rows(), q());
                U_.block(0, 0, A_.rows() / 2, q()) = PsiTD() * W() * X();
                V_ = DMatrix<double>::Zero(q(), A_.rows());
                V_.block(0, 0, q(), A_.rows() / 2) = X().transpose() * W() * Psi();
                // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from NLA module
                sol = SMW<>().solve(invA_, U_, XtWX(), V_, b_);
                // store result of smoothing
                f_.block(n_basis()*k, 0, n_basis(), 1) = sol.head(A_.rows() / 2);
                beta_ = invXtWX().solve(X().transpose() * W()) * (y() - Psi() * f_);
            }
            // store PDE misfit
            g_.block(n_basis()*k, 0, n_basis(), 1) = sol.tail(A_.rows() / 2);
        }
        return;
    }
    // getters
    SparseBlockMatrix<double, 2, 2>& A() { return A_; }
    const fdapde::SparseLU<SpMatrix<double>>& invA() const { return invA_; }
    DVector<double>& b() { return b_; }
    double norm(const DMatrix<double>& op1, const DMatrix<double>& op2) const {   // euclidian norm of op1 - op2
        return (op1 - op2).squaredNorm(); // NB: to check, defined just for compiler
    }
};

}   // namespace models
}   // namespace fdapde

#endif   // __STRPDE_NONLINEAR_H__
