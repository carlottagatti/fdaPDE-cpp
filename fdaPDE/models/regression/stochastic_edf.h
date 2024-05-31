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

#ifndef __STOCHASTIC_EDF_H__
#define __STOCHASTIC_EDF_H__

#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/utils.h>
#include "regression_type_erasure.h"

#include <random>
using fdapde::core::SMW;

namespace fdapde {
namespace models {
  
// computes an approximation of the trace of S = \Psi*T^{-1}*\Psi^T*Q using a monte carlo approximation.
class StochasticEDF {
   private:
    RegressionView<void> model_;
    std::size_t r_ = 100;   // number of monte carlo realizations
    DMatrix<double> Us_;    // sample from Rademacher distribution
    DMatrix<double> Bs_;    // \Psi^T*Q*Us_
    DMatrix<double> Y_;     // Us_^T*\Psi
    DMatrix<double> dirichlet_boundary_data_; // Dirichlet bcs in the case model_ = regression + pde regularization
    DMatrix<int> matrix_dofs_dirichlet_;      // matrix that denotes the Dirichlet boundary nodes in the case model_ = regression + pde regularization
    int m_=1;                                 // number of timesteps in the case model_ = regression + pde regularization
    int seed_ = fdapde::random_seed;
    bool init_ = false;
   public:
    // constructor
    StochasticEDF(std::size_t r, int seed) :
        r_(r), seed_((seed == fdapde::random_seed) ? std::random_device()() : seed) { }
    StochasticEDF(std::size_t r) : StochasticEDF(r, std::random_device()()) { }
    StochasticEDF() : StochasticEDF(100) { }
    StochasticEDF(std::size_t r, int seed, DMatrix<double> dirichlet_boundary_data, DMatrix<int> matrix_dofs_dirichlet, int m=1) :
        r_(r), seed_((seed == fdapde::random_seed) ? std::random_device()() : seed),
        dirichlet_boundary_data_(dirichlet_boundary_data), matrix_dofs_dirichlet_(matrix_dofs_dirichlet), m_(m) { }
  
    // evaluate trace of S exploiting a monte carlo approximation
    double compute() {
        /* std::cout << "size dirichlet_boundary_data_ = " << dirichlet_boundary_data_.rows() << "x" << dirichlet_boundary_data_.cols() << std::endl;
        std::cout << "size matrix_dofs_dirichlet_ = " << matrix_dofs_dirichlet_.rows() << "x" << matrix_dofs_dirichlet_.cols() << std::endl; */
        if (!init_) {
            // compute sample from Rademacher distribution
            std::mt19937 rng(seed_);
            std::bernoulli_distribution Be(0.5);   // bernulli distribution with parameter p = 0.5
            int space_n = model_.n_locs();
            std::cout << model_.n_locs() << " vs " << space_n << std::endl;
            Us_.resize(space_n, r_);       // preallocate memory for matrix Us
            // fill matrix
            for (std::size_t i = 0; i < space_n; ++i) {
                for (std::size_t j = 0; j < r_; ++j) {
                    if (Be(rng))
                        Us_(i, j) = 1.0;
                    else
                        Us_(i, j) = -1.0;
                }
            }
            // DMatrix<double> Im = DMatrix<double>::Ones(m_, 1);
            // Us_ = Kronecker(Im, Us_);
            // prepare matrix Y
            Y_ = Us_.transpose() * model_.Psi();
            init_ = true;   // never reinitialize again
        }
        // std::cout << "Y size = " << Y_.rows() << "x" << Y_.cols() << std::endl;
        // prepare matrix Bs_
        //std::size_t n = model_.n_basis();
        std::size_t n = model_.Psi().cols();
        Bs_ = DMatrix<double>::Zero(2 * n, r_);
        std::cout << model_.n_basis() << std::endl;
        // std::cout << "lmbq us size = " << model_.lmbQ(Us_).rows() << "x" << model_.lmbQ(Us_).cols() << std::endl;
        if (!model_.has_covariates())   // non-parametric model
            Bs_.topRows(n) = -model_.PsiTD() * model_.W() * Us_;
        else   // semi-parametric model
            Bs_.topRows(n) = -model_.PsiTD() * model_.lmbQ(Us_);
        // set Dirichlet bcs to each column of Bs_
        if (is_empty(dirichlet_boundary_data_)) std::cout << "EMPTY!" << std::endl;
        if (!is_empty(dirichlet_boundary_data_)) {
            std::size_t nn = matrix_dofs_dirichlet_.rows();

            for (size_t k=0; k<r_; ++k) {
                for (std::size_t i = 0; i < nn; ++i) {
                    if (matrix_dofs_dirichlet_(i,0) == 1) {
                        for (std::size_t j = 0; j < m_; ++j){
                            Bs_(i + j*nn, k) = dirichlet_boundary_data_(i + j*nn, 0);   // impose boundary value to the k-th column of Bs_

                            Y_(k, i + j*nn) = 1;
                        }
                    }
                }
            }
        }

        DMatrix<double> sol;              // room for problem solution
        if (!model_.has_covariates()) {   // nonparametric case
            sol = model_.invA().solve(Bs_);
        } else {
            // solve system (A+UCV)*x = Bs via woodbury decomposition using matrices U and V cached by model_
            sol = SMW<>().solve(model_.invA(), model_.U(), model_.XtWX(), model_.V(), Bs_);
        }
        // compute approximated Tr[S] using monte carlo mean
        double MCmean = 0;
        for (std::size_t i = 0; i < r_; ++i) MCmean += Y_.row(i).dot(sol.col(i).head(n));
        return MCmean / r_;
    }
    // setter
    void set_model(RegressionView<void> model) { model_ = model; }
    void set_seed(int seed) { seed_ = seed; }
    void set_n_mc_samples(int r) { r_ = r; }
};

}   // namespace models
}   // namespace fdapde

#endif   // __STOCHASTIC_EDF_H__
