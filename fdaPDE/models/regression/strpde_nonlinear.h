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
/* template <>
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

    // redefine SpaceTimeParabolicBase properties affected by nonlinear approach
    void tensorize_psi() { return; } // avoid tensorization of \Psi matrix
    void init_regularization() {
        pde_.init();
        s_ = pde_.initial_condition();   // derive initial condition from parabolic problem
        DeltaT_ = time_[1] - time_[0];   // time step (assuming equidistant points)

        // assemble matrix associated with derivation in time L_
        // [L_]_{ii} = 1/DeltaT for i \in {1 ... m} and [L_]_{i,i-1} = -1/DeltaT for i \in {1 ... m-1}
        std::size_t m_ = time_.rows();   // number of time points
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
        R0_ = pde_.mass();
        R1_ = pde_.stiff();
        R0_robin_ = pde_.mass_robin();
        // correct first n rows of discretized force as (u_1 + R0*s/DeltaT)
        u_ = pde_.force();
        u_.block(0, 0, model().n_basis(), 1) += (1.0 / DeltaT_) * (pde_.mass() * s_);
    }
    // internal utilities
    DMatrix<double> y(std::size_t k) const { return y().block(n_spatial_locs() * k, 0, n_spatial_locs(), 1); }
    DMatrix<double> u(std::size_t k) const { return u_.block(n_basis() * k, 0, n_basis(), 1); }
    DMatrix<double> u_neumann(std::size_t k) const { return pde().force_neumann().block(n_basis() * k, 0, n_basis(), 1); }
    DMatrix<double> u_robin(std::size_t k) const { return pde().force_robin().block(n_basis() * k, 0, n_basis(), 1); }
    // const SpMatrix<double>& R0() const { return pde_.mass(); }    // mass matrix in space
    // const SpMatrix<double>& R1() const { return pde_.stiff(); }   // discretization of differential operator L
    // const SpMatrix<double>& R0_robin() const { return pde_.mass_robin(); }   // mass boundary matrix due to Robin bcs

    void init_model() {   // update model object in case of **structural** changes in its definition
        // assemble system matrix for the nonparameteric part of the model
        if (is_empty(L_)) L_ = Kronecker(L(), pde().mass());
        std::cout << "begin of init_model()" << std::endl;
        std::cout << s_.rows() << " " << s_.cols() << std::endl;
        auto J = R1_step(pde().initial_condition());
        std::cout << "called R1_steps" << std::endl;
        std::cout << "size psiTD: " << PsiTD().rows() << "x" << PsiTD().cols() << std::endl;
        std::cout << "size W: " << W().rows() << "x" << W().cols() << std::endl;
        std::cout << "size psi: " << Psi().rows() << "x" << Psi().cols() << std::endl;
        std::cout << "size R1: " << J.rows() << "x" << J.cols() << std::endl;
        A_ = SparseBlockMatrix<double, 2, 2>(
          -PsiTD() * W() * Psi(), lambda_D() * (R1_step(pde().initial_condition()) + R0_robin() + lambda_T() * L_).transpose(),
          lambda_D() * (R1_step(pde().initial_condition()) + R0_robin() + lambda_T() * L_), lambda_D() * R0());
        std::cout << "DEBUG" << std::endl;
        // cache system matrix for reuse
        invA_.compute(A_);
        std::cout << "DEBUG" << std::endl;
        // prepare rhs of linear system
        b_.resize(A_.rows());
        std::cout << "DEBUG" << std::endl;
        b_.block(A_.rows() / 2, 0, A_.rows() / 2, 1) = lambda_D() * (u(0) + u_neumann(0) + u_robin(0));
        std::cout << "DEBUG" << std::endl;
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
}; */


// implementation of STRPDE for parabolic nonlinear space-time regularization, iterative approach
template <>
class STRPDE_NonLinear<SpaceTimeParabolic, iterative> :
    public RegressionBase<STRPDE_NonLinear<SpaceTimeParabolic, iterative>, SpaceTimeParabolic> {
   private:
    SparseBlockMatrix<double, 2, 2> A_ {};      // system matrix of non-parametric problem (2N x 2N matrix)
    fdapde::SparseLU<SpMatrix<double>> invA_;   // factorization of matrix A
    DVector<double> b_ {};                      // right hand side of problem's linear system (1 x 2N vector)

    // the functional minimized by the iterative scheme
    // J(f,g) = \sum_{k=1}^m (z^k - \Psi*f^k)^T*(z^k - \Psi*f^k) + \lambda_S*(g^k)^T*(g^k)
    double J(const DMatrix<double>& f, const DMatrix<double>& g) const {
        double SSE = 0;
        // SSE = \sum_{k=1}^m (z^k - \Psi*f^k)^T*(z^k - \Psi*f^k)
        for (std::size_t t = 0; t < n_temporal_locs(); ++t) {
            SSE += (y(t) - Psi() * f.block(n_spatial_basis() * t, 0, n_spatial_basis(), 1)).squaredNorm();
        }
        return SSE + lambda_D() * g.squaredNorm();
    }
    // internal solve routine used by the iterative method
    void solve(std::size_t t, BlockVector<double>& f_new, BlockVector<double>& g_new) const {
        DVector<double> x = invA_.solve(b_);
        f_new(t) = x.topRows(n_spatial_basis());
        g_new(t) = x.bottomRows(n_spatial_basis());
        return;
    }
    // internal utilities
    DMatrix<double> y(std::size_t k) const { return y().block(n_spatial_locs() * k, 0, n_spatial_locs(), 1); }
    DMatrix<double> u(std::size_t k) const { return u_.block(n_basis() * k, 0, n_basis(), 1); }
    DMatrix<double> u_neumann(std::size_t k) const { return pde().force_neumann().block(n_basis() * k, 0, n_basis(), 1); }
    DMatrix<double> u_robin(std::size_t k) const { return pde().force_robin().block(n_basis() * k, 0, n_basis(), 1); }

    // quantities related to iterative scheme
    double tol_ = 1e-4;           // tolerance used as stopping criterion
    std::size_t max_iter_ = 50;   // maximum number of allowed iterations
   public:
    using RegularizationType = SpaceTimeParabolic;
    using Base = RegressionBase<STRPDE_NonLinear<RegularizationType, iterative>, RegularizationType>;
    // import commonly defined symbols from base
    IMPORT_REGRESSION_SYMBOLS;
    using Base::DeltaT;            // distance between two time instants
    using Base::lambda_D;          // smoothing parameter in space
    using Base::lambda_T;          // smoothing parameter in time
    using Base::n_temporal_locs;   // number of time instants m defined over [0,T]
    using Base::pde_;              // parabolic differential operator df/dt + Lf - u
    // constructor
    STRPDE_NonLinear() = default;
    STRPDE_NonLinear(const pde_ptr& pde, Sampling s) : Base(pde, s) { pde_.init(); };

    // redefine SpaceTimeParabolicBase properties affected by iterative approach
    void tensorize_psi() { return; } // avoid tensorization of \Psi matrix
    void init_regularization() {
        pde_.init();
	s_ = pde_.initial_condition();
        // compute time step (assuming equidistant points)
        DeltaT_ = time_[1] - time_[0];
        u_ = pde_.force();   // compute forcing term
        // correct first n rows of discretized force as (u_1 + R0*s/DeltaT)
        u_.block(0, 0, n_basis(), 1) += (1.0 / DeltaT_) * (pde_.mass() * s_);
    }
    // getters
    const SpMatrix<double>& R0() const { return pde_.mass(); }    // mass matrix in space
    const SpMatrix<double>& R1() const { return pde_.stiff(); }   // discretization of differential operator L
    const SpMatrix<double>& R0_robin() const { return pde_.mass_robin(); }   // mass bpundary matrix due to RObin bcs
    std::size_t n_basis() const { return pde_.n_dofs(); }         // number of basis functions

    void init_model() { return; };
    void solve() {
        fdapde_assert(y().rows() != 0);
        // compute starting point (f^(k,0), g^(k,0)) k = 1 ... m for iterative minimization of functional J(f,g)
        A_ = SparseBlockMatrix<double, 2, 2>(
            PsiTD() * Psi(),   lambda_D() * (R1_step(s_) + R0_robin()).transpose(),
	        lambda_D() * (R1_step(s_) + R0_robin()), -lambda_D() * R0()           );
        // cache system matrix and its factorization
        invA_.compute(A_);
        b_.resize(A_.rows());

        // compute f^(k,0), k = 1 ... m as solution of Ax = b_(k)
        BlockVector<double> f_old(n_temporal_locs(), n_spatial_basis());
        // solve n_temporal_locs() space only linear systems
        for (std::size_t t = 0; t < n_temporal_locs(); ++t) {
            // right hand side at time step t
            b_ << PsiTD() * y(t),   // should put W()
              lambda_D() * lambda_T() * (u(t) + u_neumann(t) + u_robin(t));
            set_dirichlet_bc(A_, b_, t);
            invA_.compute(A_);
            // solve linear system Ax = b_(t) and store estimate of spatial field
            f_old(t) = invA_.solve(b_).head(A_.rows() / 2);
            // re-define matrix A
            A_ = SparseBlockMatrix<double, 2, 2>(
                PsiTD() * Psi(),   lambda_D() * (R1_step(f_old(t)) + R0_robin()).transpose() ,
	            lambda_D() * (R1_step(f_old(t)) + R0_robin()), -lambda_D() * R0()           );
            invA_.compute(A_);
        }

        // compute g^(k,0), k = 1 ... m as solution of the system
        //    G0 = [(\lambda_S*\lambda_T)/DeltaT * R_0 + \lambda_S*R_1^T]
        //    G0*g^(k,0) = \Psi^T*y^k + (\lambda_S*\lambda_T/DeltaT*R_0)*g^(k+1,0) - \Psi^T*\Psi*f^(k,0)
        SpMatrix<double> G0 =
          (lambda_D() * lambda_T() / DeltaT()) * R0() + SpMatrix<double>((lambda_D() * (R1_step(f_old(n_temporal_locs() - 1)) + R0_robin())).transpose());
        Eigen::SparseLU<SpMatrix<double>, Eigen::COLAMDOrdering<int>> invG0;
        invG0.compute(G0);   // compute factorization of matrix G0

        BlockVector<double> g_old(n_temporal_locs(), n_spatial_basis());
        // solve n_temporal_locs() distinct problems (in backward order)
        // at last step g^(t+1,0) is zero
        b_ = PsiTD() * (y(n_temporal_locs() - 1) - Psi() * f_old(n_temporal_locs() - 1));
        g_old(n_temporal_locs() - 1) = invG0.solve(b_);
        // general step
        for (int t = n_temporal_locs() - 2; t >= 0; --t) {
            // re-define matrix G0
            G0 = (lambda_D() * lambda_T() / DeltaT()) * R0() + SpMatrix<double>((lambda_D() * (R1_step(f_old(t)) + R0_robin())).transpose());
            invG0.compute(G0);
            // compute rhs at time t: \Psi^T*y^t + (\lambda_S*\lambda_T/DeltaT*R_0)*g^(t+1,0) - \Psi^T*\Psi*f^(t,0)
            b_ = PsiTD() * (y(t) - Psi() * f_old(t)) + (lambda_D() * lambda_T() / DeltaT()) * R0() * g_old(t + 1);
            // solve linear system G0*g^(t,1) = b_t and store estimate of PDE misfit
            g_old(t) = invG0.solve(b_);
        }

        // initialize value of functional J to minimize
        double Jold = std::numeric_limits<double>::max();
        double Jnew = J(f_old.get(), g_old.get());
        std::size_t i = 1;   // iteration number

        // internal iteration variables
        BlockVector<double> f_new(n_temporal_locs(), n_spatial_basis()), g_new(n_temporal_locs(), n_spatial_basis());
        // iterative scheme for minimization of functional J
        while (i < max_iter_ && std::abs((Jnew - Jold) / Jnew) > tol_) {
            std::cout << "Iteration " << i << std::endl;
            A_ = SparseBlockMatrix<double, 2, 2>(
                PsiTD() * Psi(),   lambda_D() * (R1_step(s_) + R0_robin()).transpose(),
	            lambda_D() * (R1_step(s_) + R0_robin()), -lambda_D() * R0()           );
            A_.block(0, 1) += lambda_D() * lambda_T() / DeltaT() * R0();
            A_.block(1, 0) += lambda_D() * lambda_T() / DeltaT() * R0();
            invA_.compute(A_);
            b_.resize(A_.rows());
            // at step 0 f^(k-1,i-1) is zero
            b_ << PsiTD() * y(0) + (lambda_D() * lambda_T() / DeltaT()) * R0() * g_old(1), lambda_D() * (u(0) + u_neumann(0) + u_robin(0));
            set_dirichlet_bc(A_, b_, 0);
            invA_.compute(A_);
            // solve linear system
            solve(0, f_new, g_new);
            // general step
            for (std::size_t t = 1; t < n_temporal_locs() - 1; ++t) {
                // re-compute A_
                A_ = SparseBlockMatrix<double, 2, 2>(
                    PsiTD() * Psi(),   lambda_D() * (R1_step(f_new(t-1)) + R0_robin()).transpose(),
                    lambda_D() * (R1_step(f_new(t-1)) + R0_robin()), -lambda_D() * R0()           );
                A_.block(0, 1) += lambda_D() * lambda_T() / DeltaT() * R0();
                A_.block(1, 0) += lambda_D() * lambda_T() / DeltaT() * R0();
                invA_.compute(A_);
                b_.resize(A_.rows());
                // \Psi^T*y^k   + (\lambda_D*\lambda_T/DeltaT)*R_0*g^(k+1,i-1),
                // \lambda_D*u^k + (\lambda_D*\lambda_T/DeltaT)*R_0*f^(k-1,i-1)
                b_ << PsiTD() * y(t) + (lambda_D() * lambda_T() / DeltaT()) * R0() * g_old(t + 1),
                  lambda_D() * (lambda_T() / DeltaT() * R0() * f_old(t - 1) + (u(t) + u_neumann(t) + u_robin(t)));
                set_dirichlet_bc(A_, b_, t);
                invA_.compute(A_);
                // solve linear system
                solve(t, f_new, g_new);
            }
            // re-compute A_
            A_ = SparseBlockMatrix<double, 2, 2>(
                PsiTD() * Psi(),   lambda_D() * (R1_step(f_new(n_temporal_locs() - 2)) + R0_robin()).transpose(),
                lambda_D() * (R1_step(f_new(n_temporal_locs() - 2)) + R0_robin()), -lambda_D() * R0()           );
            A_.block(0, 1) += lambda_D() * lambda_T() / DeltaT() * R0();
            A_.block(1, 0) += lambda_D() * lambda_T() / DeltaT() * R0();
            invA_.compute(A_);
            b_.resize(A_.rows());
            // at last step g^(k+1,i-1) is zero
            b_ << PsiTD() * y(n_temporal_locs() - 1),
              lambda_D() * (lambda_T() / DeltaT() * R0() * f_old(n_temporal_locs() - 2) + (u(n_temporal_locs() - 1) + u_neumann(n_temporal_locs() - 1) + u_robin(n_temporal_locs() - 1)));
            set_dirichlet_bc(A_, b_, n_temporal_locs() - 1);
            invA_.compute(A_);
            // solve linear system
            solve(n_temporal_locs() - 1, f_new, g_new);
            // prepare for next iteration
            Jold = Jnew;
            f_old = f_new;
            g_old = g_new;
            Jnew = J(f_old.get(), g_old.get());
            std::cout << "J = " << Jnew << std::endl;
            i++;
        }
        // store solution
        f_ = f_old.get();
        g_ = g_old.get();
        return;
    }
    // setters
    void set_tolerance(double tol) { tol_ = tol; }
    void set_max_iter(std::size_t max_iter) { max_iter_ = max_iter; }
};

}   // namespace models
}   // namespace fdapde

#endif   // __STRPDE_NONLINEAR_H__
