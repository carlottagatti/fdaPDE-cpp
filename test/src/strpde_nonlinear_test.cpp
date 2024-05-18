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

#include <cstddef>
#include <gtest/gtest.h>   // testing framework

#include <fdaPDE/core.h>
using fdapde::core::advection;
using fdapde::core::diffusion;
using fdapde::core::dt;
using fdapde::core::FEM;
using fdapde::core::SPLINE;
using fdapde::core::bilaplacian;
using fdapde::core::laplacian;
using fdapde::core::PDE;
using fdapde::core::Mesh;
using fdapde::core::spline_order;

#include "../../fdaPDE/models/regression/strpde_nonlinear.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::STRPDE_NonLinear;
using fdapde::models::SpaceTimeSeparable;
using fdapde::models::SpaceTimeParabolic;
using fdapde::models::Sampling;

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_mtx;
using fdapde::testing::read_csv;
using fdapde::core::NonLinearReaction;
using fdapde::core::LagrangianBasis;
using fdapde::core::non_linear_op;

// test 3
//    domain:       quasicircular domain
//    sampling:     areal
//    penalization: non-costant coefficients PDE
//    covariates:   no
//    BC:           no
//    order FE:     1
//    time penalization: parabolic (monolithic solution)
TEST(strpde_nonlinear_test, noncostantcoefficientspde_nonparametric_samplingareal_parabolic_monolithic) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(10);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 0.4 * i;
    // define spatial domain
    MeshLoader<Mesh2D> domain("quasi_circle");
    // import data from files
    DMatrix<double, Eigen::RowMajor> K_data  = read_csv<double>("../data/models/strpde/2D_test3/K.csv");
    DMatrix<double, Eigen::RowMajor> b_data  = read_csv<double>("../data/models/strpde/2D_test3/b.csv");
    DMatrix<double> subdomains = read_csv<double>("../data/models/strpde/2D_test3/incidence_matrix.csv");
    DMatrix<double> y  = read_csv<double>("../data/models/strpde/2D_test3/y.csv" );
    DMatrix<double> IC = read_csv<double>("../data/models/strpde/2D_test3/IC.csv");   // initial condition
    // define regularizing PDE
    DiscretizedMatrixField<2, 2, 2> K(K_data);
    DiscretizedVectorField<2, 2> b(b_data);
    // non linear reaction h_(u)*u
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 1 - ff[0];};
    NonLinearReaction<2, LagrangianBasis<decltype(domain.mesh),1>::ReferenceBasis> non_linear_reaction(h_);
    auto L = dt<FEM>() - diffusion<FEM>(K) + advection<FEM>(b) - non_linear_op<FEM>(non_linear_reaction);;
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 6, time_mesh.rows());
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, time_mesh, L); pde.set_forcing(u);
    pde.set_initial_condition(IC);
    // define model
    double lambda_D = std::pow(0.1, 6);
    double lambda_T = std::pow(0.1, 6);
    STRPDE_NonLinear<SpaceTimeParabolic, fdapde::monolithic> model(pde, Sampling::areal);
    model.set_lambda_D(lambda_D);
    model.set_lambda_T(lambda_T);
    model.set_spatial_locations(subdomains);
    // set model's data
    BlockFrame<double, int> df;
    df.stack(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();
    // test correctness
    EXPECT_TRUE(almost_equal(model.f(), "../data/models/strpde/2D_test3/sol.mtx"));
}