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
using fdapde::testing::pi;



// test 1 - method strpde monolithic
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: laplacian
//    covariates:   no
//    BC:           Dirichlet
//    order FE:     1
//    time penalization: parabolic (monolithic solver)
//    exact solution   : (cos(pi*x)*cos(pi*y) + 2)*exp(-t)
TEST(strpde_nonlninear_test, laplacian_nonparametric_samplingatnodes_parabolic_strpde_monolithic) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(10);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 1e-4*(i+1);
    // define spatial domain
    MeshLoader<Mesh2D> domain("unit_square_coarse");
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1) ; // has all zeros
    // import data from files
    DMatrix<double> y  = read_csv<double>("../data/models/strpde_nonlinear/2D_test1_coarse/y.csv");
    DMatrix<double> IC = read_csv<double>("../data/models/strpde_nonlinear/2D_test1_coarse/IC.csv");
    // define regularizing PDE
    auto L = dt<FEM>() - laplacian<FEM>();
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, time_mesh, L, boundary_matrix);
    pde.set_initial_condition(IC);
    //set forcing
    auto forcing_expr = [](SVector<2> x, double t) -> double {
        return -4*std::exp(-t) + std::cos(pi*x[0])*std::cos(pi*x[1])*std::exp(-t)*(-2+2*pi*pi) + (4 + std::cos(pi*x[0])*std::cos(pi*x[0])*std::cos(pi*x[1])*std::cos(pi*x[1]) + 4*std::cos(pi*x[0])*std::cos(pi*x[1]))*std::exp(-2*t);
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);
    // set dirichlet bcs
    auto solution_expr = [](SVector<2> x, double t) -> double {
        return (std::cos(pi*x[0])*std::cos(pi*x[1]) + 2)*std::exp(-t);
    };
    DMatrix<double> nodes_ = pde.dof_coords();
    DMatrix<double> dirichlet_bc(nodes_.rows(), time_mesh.size());
    for (int i = 0; i < nodes_.rows(); ++i) {
        for (int j = 0; j < time_mesh.size(); ++j) {
            dirichlet_bc(i, j) = solution_expr(nodes_.row(i), time_mesh(j));
        }
    }
    pde.set_dirichlet_bc(dirichlet_bc);
    // define model
    double lambda_D = 1;
    double lambda_T = 1;
    STRPDE<SpaceTimeParabolic, fdapde::monolithic> model(pde, Sampling::mesh_nodes);
    model.set_lambda_D(lambda_D);
    model.set_lambda_T(lambda_T);
    // set model's data
    BlockFrame<double, int> df;
    df.stack(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    std::cout << "debug" << std::endl;
    model.solve();
    std::cout << "debug" << std::endl;
    DMatrix<double> sol = read_csv<double>("../data/models/strpde_nonlinear/2D_test1_coarse/sol.csv");
    
    // test corretness
    double MSE = 0;
    int n = model.f().rows();
    int m = time_mesh.rows();
    for (size_t i=0; i<n; ++i)
    {
        MSE += (sol(i) - model.f()(i))*(sol(i) - model.f()(i)) / (n);
    }
    std::cout << "MSE = " << MSE << std::endl;

    // EXPECT_TRUE(almost_equal(model.f(), sol));
    std::cout << (model.f()-sol).lpNorm<Eigen::Infinity>() << std::endl;
    EXPECT_TRUE(almost_equal(model.f(), sol));
}

// test 1 - method strpde iterative
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: laplacian
//    covariates:   no
//    BC:           Dirichlet
//    order FE:     1
//    time penalization: parabolic (itervative solver)
//    exact solution   : (cos(pi*x)*cos(pi*y) + 2)*exp(-t)
TEST(strpde_nonlninear_test, laplacian_nonparametric_samplingatnodes_parabolic_strpde_iterative) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(10);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 1e-4*(i+1);
    // define spatial domain
    MeshLoader<Mesh2D> domain("unit_square_coarse");
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1) ; // has all zeros
    // import data from files
    DMatrix<double> y  = read_csv<double>("../data/models/strpde_nonlinear/2D_test1_coarse/y.csv");
    DMatrix<double> IC = read_csv<double>("../data/models/strpde_nonlinear/2D_test1_coarse/IC.csv");
    // define regularizing PDE
    auto L = dt<FEM>() - laplacian<FEM>();
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, time_mesh, L, boundary_matrix);
    pde.set_initial_condition(IC);
    //set forcing
    auto forcing_expr = [](SVector<2> x, double t) -> double {
        return -4*std::exp(-t) + std::cos(pi*x[0])*std::cos(pi*x[1])*std::exp(-t)*(-2+2*pi*pi) + (4 + std::cos(pi*x[0])*std::cos(pi*x[0])*std::cos(pi*x[1])*std::cos(pi*x[1]) + 4*std::cos(pi*x[0])*std::cos(pi*x[1]))*std::exp(-2*t);
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);
    // set dirichlet bcs
    auto solution_expr = [](SVector<2> x, double t) -> double {
        return (std::cos(pi*x[0])*std::cos(pi*x[1]) + 2)*std::exp(-t);
    };
    DMatrix<double> nodes_ = pde.dof_coords();
    DMatrix<double> dirichlet_bc(nodes_.rows(), time_mesh.size());
    for (int i = 0; i < nodes_.rows(); ++i) {
        for (int j = 0; j < time_mesh.size(); ++j) {
            dirichlet_bc(i, j) = solution_expr(nodes_.row(i), time_mesh(j));
        }
    }
    pde.set_dirichlet_bc(dirichlet_bc);
    // define model
    double lambda_D = 1;
    double lambda_T = 1;
    STRPDE<SpaceTimeParabolic, fdapde::iterative> model(pde, Sampling::mesh_nodes);
    model.set_lambda_D(lambda_D);
    model.set_lambda_T(lambda_T);
    // set model's data
    BlockFrame<double, int> df;
    df.stack(OBSERVATIONS_BLK, y);
    model.set_data(df);
    model.set_tolerance(1e-5);
    // solve smoothing problem
    model.init();
    model.solve();
    DMatrix<double> sol = read_csv<double>("../data/models/strpde_nonlinear/2D_test1_coarse/sol.csv");
    
    // test corretness
    // create a grid of points different from the sampling points where we can compute the MSE
    STRPDE<SpaceTimeParabolic, fdapde::iterative> model_grid(pde, Sampling::pointwise);
    DMatrix<double> grid = read_csv<double>("../data/models/strpde_nonlinear/2D_test1_coarse/grid_locs.csv");
    model_grid.set_spatial_locations(grid);
    model_grid.init_sampling(true);  // evalute Psi() in the new grid
    // compute MSE over the grid
    DVector<double> f_grid(grid.rows(), 1);
    int m = time_mesh.rows();
    SpMatrix<double> Im(m,m);
    Im.setIdentity();
    f_grid = Kronecker(Im,model_grid.Psi())*model.f();
    DMatrix<double> sol_grid = read_csv<double>("../data/models/strpde_nonlinear/2D_test1_coarse/sol_grid.csv");
    double MSE_grid = 0;
    int n = f_grid.rows();
    for (size_t i=0; i<n; ++i)
    {
        MSE_grid += (sol_grid(i) - f_grid(i))*(sol_grid(i) - f_grid(i)) / (n);
    }
    std::cout << "MSE_grid = " << MSE_grid << std::endl;
    std::cout << "inf_norm error in sgrid points = " << (f_grid - sol_grid).lpNorm<Eigen::Infinity>() << std::endl;

    // compute MSE over the nodes
    double MSE = 0;
    n = model.f().rows();
    for (size_t i=0; i<n; ++i)
    {
        MSE += (sol(i) - model.f()(i))*(sol(i) - model.f()(i)) / (n);
    }
    std::cout << "MSE = " << MSE << std::endl;

    // EXPECT_TRUE(almost_equal(model.f(), sol));
    std::cout << (model.f()-sol).lpNorm<Eigen::Infinity>() << std::endl;
    EXPECT_TRUE(almost_equal(model.f(), sol));
}

// test 1 - method iterative_EI
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: laplacian + nonlinear reaction
//    covariates:   no
//    BC:           Dirichlet
//    order FE:     1
//    time penalization: parabolic (iterative_EI solver)
//    exact solution   : (cos(pi*x)*cos(pi*y) + 2)*exp(-t)
TEST(strpde_nonlninear_test, laplacian_nonparametric_samplingatnodes_parabolic_iterative_EI) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(10);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 1e-4*(i+1);
    // define spatial domain
    MeshLoader<Mesh2D> domain("unit_square_coarse");
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1) ; // has all zeros
    // import data from files
    DMatrix<double> y  = read_csv<double>("../data/models/strpde_nonlinear/2D_test1_coarse/y.csv");
    DMatrix<double> IC = read_csv<double>("../data/models/strpde_nonlinear/2D_test1_coarse/IC.csv");
    // define regularizing PDE
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 1 - ff[0];};
    NonLinearReaction<2, LagrangianBasis<decltype(domain.mesh),1>::ReferenceBasis> non_linear_reaction(h_);
    auto L = dt<FEM>() - laplacian<FEM>() - non_linear_op<FEM>(non_linear_reaction);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, time_mesh, L, h_, boundary_matrix);
    pde.set_initial_condition(IC);
    //set forcing
    auto forcing_expr = [](SVector<2> x, double t) -> double {
        return -4*std::exp(-t) + std::cos(pi*x[0])*std::cos(pi*x[1])*std::exp(-t)*(-2+2*pi*pi) + (4 + std::cos(pi*x[0])*std::cos(pi*x[0])*std::cos(pi*x[1])*std::cos(pi*x[1]) + 4*std::cos(pi*x[0])*std::cos(pi*x[1]))*std::exp(-2*t);
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);
    // set dirichlet bcs
    auto solution_expr = [](SVector<2> x, double t) -> double {
        return (std::cos(pi*x[0])*std::cos(pi*x[1]) + 2)*std::exp(-t);
    };
    DMatrix<double> nodes_ = pde.dof_coords();
    DMatrix<double> dirichlet_bc(nodes_.rows(), time_mesh.size());
    for (int i = 0; i < nodes_.rows(); ++i) {
        for (int j = 0; j < time_mesh.size(); ++j) {
            dirichlet_bc(i, j) = solution_expr(nodes_.row(i), time_mesh(j));
        }
    }
    pde.set_dirichlet_bc(dirichlet_bc);
    // define model
    double lambda_D = 1;
    double lambda_T = 1;
    STRPDE_NonLinear<SpaceTimeParabolic, fdapde::iterative_EI> model(pde, Sampling::mesh_nodes);
    model.set_lambda_D(lambda_D);
    model.set_lambda_T(lambda_T);
    // set model's data
    BlockFrame<double, int> df;
    df.stack(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // set parameters for iterative method
    model.set_tolerance(1e-5, 1e-8);
    model.set_max_iter(50);
    // solve smoothing problem
    model.init();
    model.solve();
    DMatrix<double> sol = read_csv<double>("../data/models/strpde_nonlinear/2D_test1_coarse/sol.csv");
    
    // test corretness
    std::cout << "size psi = " << model.Psi().rows() << "x" << model.Psi().cols() << std::endl;
    std::cout << "size f = " << model.f().rows() << "x" << model.f().cols() << std::endl;
    double MSE = 0;
    int n = model.f().rows();
    int m = time_mesh.rows();
    std::cout << "n normale = " << n << std::endl;
    for (size_t i=0; i<n; ++i)
    {
        MSE += (sol(i) - model.f()(i))*(sol(i) - model.f()(i)) / (n);
    }
    std::cout << "MSE = " << MSE << std::endl;

    // EXPECT_TRUE(almost_equal(model.f(), sol));
    std::cout << (model.f()-sol).lpNorm<Eigen::Infinity>() << std::endl;
    EXPECT_TRUE(almost_equal(model.f(), sol));
}

// test 1 - method iterative
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: laplacian + nonlinear reaction
//    covariates:   no
//    BC:           Dirichlet
//    order FE:     1
//    time penalization: parabolic (iterative solver)
//    exact solution   : (cos(pi*x)*cos(pi*y) + 2)*exp(-t)
TEST(strpde_nonlninear_test, laplacian_nonparametric_samplingatnodes_parabolic_iterative) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(10);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 1e-4*(i+1);
    // define spatial domain
    MeshLoader<Mesh2D> domain("unit_square_coarse");
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1) ; // has all zeros
    // import data from files
    DMatrix<double> y  = read_csv<double>("../data/models/strpde_nonlinear/2D_test1_coarse/y.csv");
    DMatrix<double> IC = read_csv<double>("../data/models/strpde_nonlinear/2D_test1_coarse/IC.csv");
    // define regularizing PDE
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 1 - ff[0];};
    NonLinearReaction<2, LagrangianBasis<decltype(domain.mesh),1>::ReferenceBasis> non_linear_reaction(h_);
    auto L = dt<FEM>() - laplacian<FEM>() - non_linear_op<FEM>(non_linear_reaction);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, time_mesh, L, h_, boundary_matrix);
    pde.set_initial_condition(IC);
    //set forcing
    auto forcing_expr = [](SVector<2> x, double t) -> double {
        return -4*std::exp(-t) + std::cos(pi*x[0])*std::cos(pi*x[1])*std::exp(-t)*(-2+2*pi*pi) + (4 + std::cos(pi*x[0])*std::cos(pi*x[0])*std::cos(pi*x[1])*std::cos(pi*x[1]) + 4*std::cos(pi*x[0])*std::cos(pi*x[1]))*std::exp(-2*t);
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);
    // set dirichlet bcs
    auto solution_expr = [](SVector<2> x, double t) -> double {
        return (std::cos(pi*x[0])*std::cos(pi*x[1]) + 2)*std::exp(-t);
    };
    DMatrix<double> nodes_ = pde.dof_coords();
    DMatrix<double> dirichlet_bc(nodes_.rows(), time_mesh.size());
    for (int i = 0; i < nodes_.rows(); ++i) {
        for (int j = 0; j < time_mesh.size(); ++j) {
            dirichlet_bc(i, j) = solution_expr(nodes_.row(i), time_mesh(j));
        }
    }
    pde.set_dirichlet_bc(dirichlet_bc);
    // define model
    double lambda_D = 1;
    double lambda_T = 1;
    STRPDE_NonLinear<SpaceTimeParabolic, fdapde::iterative> model(pde, Sampling::mesh_nodes);
    model.set_lambda_D(lambda_D);
    model.set_lambda_T(lambda_T);
    // set model's data
    BlockFrame<double, int> df;
    df.stack(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // set parameters for iterative method
    model.set_tolerance(1e-5, 1e-8);
    model.set_max_iter(50);
    // solve smoothing problem
    model.init();
    model.solve();
    DMatrix<double> sol = read_csv<double>("../data/models/strpde_nonlinear/2D_test1_coarse/sol.csv");
    
    // test corretness
    double MSE = 0;
    int n = model.f().rows();
    int m = time_mesh.rows();
    for (size_t i=0; i<n; ++i)
    {
        MSE += (sol(i) - model.f()(i))*(sol(i) - model.f()(i)) / (n);
    }
    std::cout << "MSE = " << MSE << std::endl;

    // EXPECT_TRUE(almost_equal(model.f(), sol));
    std::cout << (model.f()-sol).lpNorm<Eigen::Infinity>() << std::endl;
    EXPECT_TRUE(almost_equal(model.f(), sol));
}


// test 1 - method monolithic
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: laplacian + nonlinear reaction
//    covariates:   no
//    BC:           Dirichlet
//    order FE:     1
//    time penalization: parabolic (monolithic solver)
//    exact solution   : (cos(pi*x)*cos(pi*y) + 2)*exp(-t)
TEST(strpde_nonlninear_test, laplacian_nonparametric_samplingatnodes_parabolic_monolithic) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(10);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 1e-4*(i+1);
    // define spatial domain
    MeshLoader<Mesh2D> domain("unit_square_coarse");
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1);
    // import data from files
    DMatrix<double> y  = read_csv<double>("../data/models/strpde_nonlinear/2D_test1_coarse/y.csv");
    DMatrix<double> IC = read_csv<double>("../data/models/strpde_nonlinear/2D_test1_coarse/IC.csv");
    // define regularizing PDE
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 1 - ff[0];};
    NonLinearReaction<2, LagrangianBasis<decltype(domain.mesh),1>::ReferenceBasis> non_linear_reaction(h_);
    auto L = dt<FEM>() - laplacian<FEM>() - non_linear_op<FEM>(non_linear_reaction);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, time_mesh, L, h_, boundary_matrix);
    pde.set_initial_condition(IC);
    //set forcing
    auto forcing_expr = [](SVector<2> x, double t) -> double {
        return -4*std::exp(-t) + std::cos(pi*x[0])*std::cos(pi*x[1])*std::exp(-t)*(-2+2*pi*pi) + (4 + std::cos(pi*x[0])*std::cos(pi*x[0])*std::cos(pi*x[1])*std::cos(pi*x[1]) + 4*std::cos(pi*x[0])*std::cos(pi*x[1]))*std::exp(-2*t);
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);
    // set dirichlet bcs
    auto solution_expr = [](SVector<2> x, double t) -> double {
        return (std::cos(pi*x[0])*std::cos(pi*x[1]) + 2)*std::exp(-t);
    };
    DMatrix<double> nodes_ = pde.dof_coords();
    DMatrix<double> dirichlet_bc(nodes_.rows(), time_mesh.size());
    for (int i = 0; i < nodes_.rows(); ++i) {
        for (int j = 0; j < time_mesh.size(); ++j) {
            dirichlet_bc(i, j) = solution_expr(nodes_.row(i), time_mesh(j));
        }
    }
    pde.set_dirichlet_bc(dirichlet_bc);
    // define model
    double lambda_D = 2;
    double lambda_T = 1;
    STRPDE_NonLinear<SpaceTimeParabolic, fdapde::monolithic> model(pde, Sampling::mesh_nodes);
    model.set_lambda_D(lambda_D);
    model.set_lambda_T(lambda_T);
    // set model's data
    BlockFrame<double, int> df;
    df.stack(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // set parameters for iterative method
    model.set_tolerance(1e-5);
    model.set_max_iter(15);
    // solve smoothing problem
    model.init();
    model.solve();
    DMatrix<double> sol = read_csv<double>("../data/models/strpde_nonlinear/2D_test1_coarse/sol.csv");
    
    // test corretness
    double MSE = 0;
    int n = model.f().rows();
    int m = time_mesh.rows();
    for (size_t i=0; i<n; ++i)
    {
        MSE += (sol(i) - model.f()(i))*(sol(i) - model.f()(i)) / (n);
    }
    std::cout << "MSE = " << MSE << std::endl;

    // EXPECT_TRUE(almost_equal(model.f(), sol));
    std::cout << (model.f()-sol).lpNorm<Eigen::Infinity>() << std::endl;
    EXPECT_TRUE(almost_equal(model.f(), sol));
}

// test 2 - method iterative_EI
//    domain:       unit square [1,1] x [1,1]
//    sampling:     space locations != nodes
//    penalization: laplacian + nonlinear reaction
//    covariates:   no
//    BC:           Dirichlet
//    order FE:     1
//    time penalization: parabolic (iterative_EI solver)
//    exact solution   : (cos(pi*x)*cos(pi*y) + 2)*exp(-t)
TEST(strpde_nonlninear_test, laplacian_nonparametric_sapcesamplingpoitwise_parabolic_iterative_EI) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(10);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 1e-4*(i+1);
    // define spatial domain
    MeshLoader<Mesh2D> domain("unit_square_coarse");
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1) ; // has all zeros
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/strpde_nonlinear/2D_test2_coarse_sampling40space/space_locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/strpde_nonlinear/2D_test2_coarse_sampling40space/y.csv");
    DMatrix<double> IC   = read_csv<double>("../data/models/strpde_nonlinear/2D_test2_coarse_sampling40space/IC.csv");
    // define regularizing PDE
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 1 - ff[0];};
    NonLinearReaction<2, LagrangianBasis<decltype(domain.mesh),1>::ReferenceBasis> non_linear_reaction(h_);
    auto L = dt<FEM>() - laplacian<FEM>() - non_linear_op<FEM>(non_linear_reaction);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, time_mesh, L, h_, boundary_matrix);
    pde.set_initial_condition(IC);
    //set forcing
    auto forcing_expr = [](SVector<2> x, double t) -> double {
        return -4*std::exp(-t) + std::cos(pi*x[0])*std::cos(pi*x[1])*std::exp(-t)*(-2+2*pi*pi) + (4 + std::cos(pi*x[0])*std::cos(pi*x[0])*std::cos(pi*x[1])*std::cos(pi*x[1]) + 4*std::cos(pi*x[0])*std::cos(pi*x[1]))*std::exp(-2*t);
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);
    // set dirichlet bcs
    auto solution_expr = [](SVector<2> x, double t) -> double {
        return (std::cos(pi*x[0])*std::cos(pi*x[1]) + 2)*std::exp(-t);
    };
    DMatrix<double> nodes_ = pde.dof_coords();
    DMatrix<double> dirichlet_bc(nodes_.rows(), time_mesh.size());
    for (int i = 0; i < nodes_.rows(); ++i) {
        for (int j = 0; j < time_mesh.size(); ++j) {
            dirichlet_bc(i, j) = solution_expr(nodes_.row(i), time_mesh(j));
        }
    }
    pde.set_dirichlet_bc(dirichlet_bc);
    // define model
    double lambda_D = 1;
    double lambda_T = 1;
    STRPDE_NonLinear<SpaceTimeParabolic, fdapde::iterative_EI> model(pde, Sampling::pointwise);
    model.set_lambda_D(lambda_D);
    model.set_lambda_T(lambda_T);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
    df.stack(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // set parameters for iterative method
    model.set_tolerance(1e-5);
    model.set_max_iter(50);
    // solve smoothing problem
    std::cout << "debug" << std::endl;
    model.init();
    std::cout << "debug" << std::endl;
    model.solve();
    std::cout << "debug" << std::endl;
    DMatrix<double> sol = read_csv<double>("../data/models/strpde_nonlinear/2D_test2_coarse_sampling40space/sol.csv");
    
    // test corretness
    std::cout << "debug" << std::endl;
    DVector<double> f_in_sampling_points(locs.rows(), 1);
    f_in_sampling_points = model.Psi()*model.f();
    DMatrix<double> sol_in_sampling_points = read_csv<double>("../data/models/strpde_nonlinear/2D_test2_coarse_sampling40space/sol_samplingpoints.csv");
    double MSE_sampling = 0;
    int n = f_in_sampling_points.rows();
    int m = time_mesh.rows();
    for (size_t i=0; i<n; ++i)
    {
        MSE_sampling += (sol_in_sampling_points(i) - f_in_sampling_points(i))*(sol_in_sampling_points(i) - f_in_sampling_points(i)) / (n);
    }
    std::cout << "MSE_sampling = " << MSE_sampling << std::endl;
    std::cout << "inf_norm error in sampling points = " << (f_in_sampling_points-sol_in_sampling_points).lpNorm<Eigen::Infinity>() << std::endl;

    double MSE = 0;
    n = model.f().rows();
    for (size_t i=0; i<n; ++i)
    {
        MSE += (sol(i) - model.f()(i))*(sol(i) - model.f()(i)) / (n);
    }
    std::cout << "MSE = " << MSE << std::endl;

    // EXPECT_TRUE(almost_equal(model.f(), sol));
    std::cout << (model.f()-sol).lpNorm<Eigen::Infinity>() << std::endl;
    EXPECT_TRUE(almost_equal(model.f(), sol));
}

// test 2 - method iterative
//    domain:       unit square [1,1] x [1,1]
//    sampling:     space locations != nodes
//    penalization: laplacian + nonlinear reaction
//    covariates:   no
//    BC:           Dirichlet
//    order FE:     1
//    time penalization: parabolic (iterative solver)
//    exact solution   : (cos(pi*x)*cos(pi*y) + 2)*exp(-t)
TEST(strpde_nonlninear_test, laplacian_nonparametric_sapcesamplingpoitwise_parabolic_iterative) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(10);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 1e-4*(i+1);
    // define spatial domain
    MeshLoader<Mesh2D> domain("unit_square_coarse");
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1) ; // has all zeros
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/strpde_nonlinear/2D_test2_coarse_sampling40space/space_locs.csv");
    DMatrix<double> y  = read_csv<double>("../data/models/strpde_nonlinear/2D_test2_coarse_sampling40space/y.csv");
    DMatrix<double> IC = read_csv<double>("../data/models/strpde_nonlinear/2D_test2_coarse_sampling40space/IC.csv");
    // define regularizing PDE
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 1 - ff[0];};
    NonLinearReaction<2, LagrangianBasis<decltype(domain.mesh),1>::ReferenceBasis> non_linear_reaction(h_);
    auto L = dt<FEM>() - laplacian<FEM>() - non_linear_op<FEM>(non_linear_reaction);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, time_mesh, L, h_, boundary_matrix);
    pde.set_initial_condition(IC);
    //set forcing
    auto forcing_expr = [](SVector<2> x, double t) -> double {
        return -4*std::exp(-t) + std::cos(pi*x[0])*std::cos(pi*x[1])*std::exp(-t)*(-2+2*pi*pi) + (4 + std::cos(pi*x[0])*std::cos(pi*x[0])*std::cos(pi*x[1])*std::cos(pi*x[1]) + 4*std::cos(pi*x[0])*std::cos(pi*x[1]))*std::exp(-2*t);
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);
    // set dirichlet bcs
    auto solution_expr = [](SVector<2> x, double t) -> double {
        return (std::cos(pi*x[0])*std::cos(pi*x[1]) + 2)*std::exp(-t);
    };
    DMatrix<double> nodes_ = pde.dof_coords();
    DMatrix<double> dirichlet_bc(nodes_.rows(), time_mesh.size());
    for (int i = 0; i < nodes_.rows(); ++i) {
        for (int j = 0; j < time_mesh.size(); ++j) {
            dirichlet_bc(i, j) = solution_expr(nodes_.row(i), time_mesh(j));
        }
    }
    pde.set_dirichlet_bc(dirichlet_bc);
    // define model
    double lambda_D = 1;
    double lambda_T = 1;
    STRPDE_NonLinear<SpaceTimeParabolic, fdapde::iterative> model(pde, Sampling::pointwise);
    model.set_lambda_D(lambda_D);
    model.set_lambda_T(lambda_T);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
    df.stack(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // set parameters for iterative method
    model.set_tolerance(1e-5);
    model.set_max_iter(50);
    // solve smoothing problem
    std::cout << "debug" << std::endl;
    model.init();
    std::cout << "debug" << std::endl;
    model.solve();
    std::cout << "debug" << std::endl;
    DMatrix<double> sol = read_csv<double>("../data/models/strpde_nonlinear/2D_test2_coarse_sampling40space/sol.csv");
    
    // test corretness
    std::cout << "debug" << std::endl;
    DVector<double> f_in_sampling_points(locs.rows(), 1);
    f_in_sampling_points = model.Psi()*model.f();
    DMatrix<double> sol_in_sampling_points = read_csv<double>("../data/models/strpde_nonlinear/2D_test2_coarse_sampling40space/sol_samplingpoints.csv");
    double MSE_sampling = 0;
    int n = f_in_sampling_points.rows();
    int m = time_mesh.rows();
    for (size_t i=0; i<n; ++i)
    {
        MSE_sampling += (sol_in_sampling_points(i) - f_in_sampling_points(i))*(sol_in_sampling_points(i) - f_in_sampling_points(i)) / (n);
    }
    std::cout << "MSE_sampling = " << MSE_sampling << std::endl;
    std::cout << "inf_norm error in sampling points = " << (f_in_sampling_points-sol_in_sampling_points).lpNorm<Eigen::Infinity>() << std::endl;

    double MSE = 0;
    n = model.f().rows();
    for (size_t i=0; i<n; ++i)
    {
        MSE += (sol(i) - model.f()(i))*(sol(i) - model.f()(i)) / (n);
    }
    std::cout << "MSE = " << MSE << std::endl;

    // EXPECT_TRUE(almost_equal(model.f(), sol));
    std::cout << (model.f()-sol).lpNorm<Eigen::Infinity>() << std::endl;
    EXPECT_TRUE(almost_equal(model.f(), sol));
}

// test 2 - method monolithic
//    domain:       unit square [1,1] x [1,1]
//    sampling:     space locations != nodes
//    penalization: laplacian + nonlinear reaction
//    covariates:   no
//    BC:           Dirichlet
//    order FE:     1
//    time penalization: parabolic (monolithic solver)
//    exact solution   : (cos(pi*x)*cos(pi*y) + 2)*exp(-t)
TEST(strpde_nonlninear_test, laplacian_nonparametric_sapcesamplingpoitwise_parabolic_monolithic) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(10);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 1e-4*(i+1);
    // define spatial domain
    MeshLoader<Mesh2D> domain("unit_square_coarse");
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1) ; // has all zeros
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/strpde_nonlinear/2D_test2_coarse_sampling40space/space_locs.csv");
    DMatrix<double> y  = read_csv<double>("../data/models/strpde_nonlinear/2D_test2_coarse_sampling40space/y.csv");
    DMatrix<double> IC = read_csv<double>("../data/models/strpde_nonlinear/2D_test2_coarse_sampling40space/IC.csv");
    // define regularizing PDE
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 1 - ff[0];};
    NonLinearReaction<2, LagrangianBasis<decltype(domain.mesh),1>::ReferenceBasis> non_linear_reaction(h_);
    auto L = dt<FEM>() - laplacian<FEM>() - non_linear_op<FEM>(non_linear_reaction);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, time_mesh, L, h_, boundary_matrix);
    pde.set_initial_condition(IC);
    //set forcing
    auto forcing_expr = [](SVector<2> x, double t) -> double {
        return -4*std::exp(-t) + std::cos(pi*x[0])*std::cos(pi*x[1])*std::exp(-t)*(-2+2*pi*pi) + (4 + std::cos(pi*x[0])*std::cos(pi*x[0])*std::cos(pi*x[1])*std::cos(pi*x[1]) + 4*std::cos(pi*x[0])*std::cos(pi*x[1]))*std::exp(-2*t);
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);
    // set dirichlet bcs
    auto solution_expr = [](SVector<2> x, double t) -> double {
        return (std::cos(pi*x[0])*std::cos(pi*x[1]) + 2)*std::exp(-t);
    };
    DMatrix<double> nodes_ = pde.dof_coords();
    DMatrix<double> dirichlet_bc(nodes_.rows(), time_mesh.size());
    for (int i = 0; i < nodes_.rows(); ++i) {
        for (int j = 0; j < time_mesh.size(); ++j) {
            dirichlet_bc(i, j) = solution_expr(nodes_.row(i), time_mesh(j));
        }
    }
    pde.set_dirichlet_bc(dirichlet_bc);
    // define model
    double lambda_D = 1;
    double lambda_T = 1;
    STRPDE_NonLinear<SpaceTimeParabolic, fdapde::monolithic> model(pde, Sampling::pointwise);
    model.set_lambda_D(lambda_D);
    model.set_lambda_T(lambda_T);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
    df.stack(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // set parameters for iterative method
    model.set_tolerance(1e-5);
    model.set_max_iter(50);
    // solve smoothing problem
    model.init();
    model.solve();
    DMatrix<double> sol = read_csv<double>("../data/models/strpde_nonlinear/2D_test2_coarse_sampling40space/sol.csv");
    
    // test corretness
    DVector<double> f_in_sampling_points(locs.rows(), 1);
    f_in_sampling_points = model.Psi()*model.f();
    DMatrix<double> sol_in_sampling_points = read_csv<double>("../data/models/strpde_nonlinear/2D_test2_coarse_sampling40space/sol_samplingpoints.csv");

    double MSE_sampling = 0;
    int n = f_in_sampling_points.rows();
    int m = time_mesh.rows();
    for (size_t i=0; i<n; ++i)
    {
        MSE_sampling += (sol_in_sampling_points(i) - f_in_sampling_points(i))*(sol_in_sampling_points(i) - f_in_sampling_points(i)) / (n);
    }
    std::cout << "MSE_sampling = " << MSE_sampling << std::endl;
    std::cout << "inf_norm error in sampling points = " << (f_in_sampling_points-sol_in_sampling_points).lpNorm<Eigen::Infinity>() << std::endl;

    double MSE = 0;
    n = model.f().rows();
    m = time_mesh.rows();
    for (size_t i=0; i<n; ++i)
    {
        MSE += (sol(i) - model.f()(i))*(sol(i) - model.f()(i)) / (n);
    }
    std::cout << "MSE = " << MSE << std::endl;
    std::cout << "inf_norm error at nodes = " << (model.f()-sol).lpNorm<Eigen::Infinity>() << std::endl;

    EXPECT_TRUE(almost_equal(model.f(), sol));
}

// test 3 - method iterative_EI
//    domain:       unit cube [0,1] x [0,1] x [0,1]
//    sampling:     locations = nodes
//    penalization: -laplacian + advection(b) + nonlinear reaction
//    covariates:   no
//    BC:           Dirichlet + Neumann
//    order FE:     1
//    time penalization: parabolic (iterative_EI solver)
//    exact solution   : (x + y^2 + z^2)*t
TEST(strpde_nonlninear_test, cube_nonparametric_samplingatnodes_parabolic_iterative_EI) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(10);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 1e-4*(i+1);
    // define spatial domain
    MeshLoader<Mesh3D> domain("unit_cube_5");
    // define the boundary with a DMatrix (=0 if Dirichlet, =1 if Neumann, =2 if Robin)
    // considering the unit cube,
    // we have Neumann boundary when z=0 and z=1 (upper and lower sides)
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1) ; // has all zeros
    for (size_t j=0; j<36; ++j) boundary_matrix(j, 0) = 1;
    for (size_t j=180; j<216; ++j) boundary_matrix(j, 0) = 1;
    // set the Dirichlet nodes at the edges
    for (size_t j=0; j<6; j++) {
        boundary_matrix(6*j, 0) = 0;      // lower face, points with x=0
        boundary_matrix(5 + j*6, 0) = 0; // lower face, points with x=1
        boundary_matrix(j, 0) = 0;         // lower face, points with y=0
        boundary_matrix(30 + j, 0) = 0;   // lower face, points with y=1

        boundary_matrix(180 + 6*j, 0) = 0; // upper face, points with x=0
        boundary_matrix(185 + 6*j, 0) = 0; // upper face, points with x=1
        boundary_matrix(180 + j, 0) = 0;    // upper face, points with y=0
        boundary_matrix(210 + j, 0) = 0;    // upper face, points with y=1
    }

    // import data from files
    DMatrix<double> y  = read_csv<double>("../data/models/strpde_nonlinear/3D_test3_cube5_samplingatnodes/y.csv");
    DMatrix<double> IC = read_csv<double>("../data/models/strpde_nonlinear/3D_test3_cube5_samplingatnodes/IC.csv");
    // define regularizing PDE
    SVector<3> b_; b_ << 2., -1., 1.;  // advection coefficient
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 1 - ff[0];};
    NonLinearReaction<3, LagrangianBasis<decltype(domain.mesh),1>::ReferenceBasis> non_linear_reaction(h_);
    auto L = dt<FEM>() -laplacian<FEM>() + advection<FEM>(b_) + non_linear_op<FEM>(non_linear_reaction);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, time_mesh, L, h_, boundary_matrix);
    pde.set_initial_condition(IC);
    //set forcing
    auto forcing_expr = [](SVector<3> x, double t) -> double {
        return x[0] + x[1]*x[1] + x[2]*x[2] - 4*t + (2 - 2*x[1] + 2*x[2])*t + (x[0] + x[1]*x[1] + x[2]*x[2])*t*(1 - (x[0] + x[1]*x[1] + x[2]*x[2])*t);
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);
    // set dirichlet bcs
    auto solution_expr = [](SVector<3> x, double t) -> double {
        return (x[0] + x[1]*x[1] + x[2]*x[2])*t;
    };
    DMatrix<double> nodes_ = pde.dof_coords();
    DMatrix<double> dirichlet_bc(nodes_.rows(), time_mesh.size());
    for (int i = 0; i < nodes_.rows(); ++i) {
        for (int j = 0; j < time_mesh.size(); ++j) {
            dirichlet_bc(i, j) = solution_expr(nodes_.row(i), time_mesh(j));
        }
    }
    pde.set_dirichlet_bc(dirichlet_bc);
    // set neumann bcs
    auto neumann_expr = [](SVector<3> x, double t) -> double { 
        return 2*x[2]*t;
    };
    DMatrix<double> boundary_quadrature_nodes = pde.boundary_quadrature_nodes();
    DMatrix<double> f_neumann(boundary_quadrature_nodes.rows(), time_mesh.size());
    for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
        for (int j = 0; j < time_mesh.size(); ++j) f_neumann(i, j) = neumann_expr(boundary_quadrature_nodes.row(i), time_mesh(j));
    }
    pde.set_neumann_bc(f_neumann);
    // define model
    double lambda_D = 1;
    double lambda_T = 1;
    STRPDE_NonLinear<SpaceTimeParabolic, fdapde::iterative_EI> model(pde, Sampling::mesh_nodes);
    model.set_lambda_D(lambda_D);
    model.set_lambda_T(lambda_T);
    // set model's data
    BlockFrame<double, int> df;
    df.stack(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // set parameters for iterative method
    model.set_tolerance(1e-5);
    model.set_max_iter(50);
    // solve smoothing problem
    model.init();
    model.solve();
    DMatrix<double> sol = read_csv<double>("../data/models/strpde_nonlinear/3D_test3_cube5_samplingatnodes/sol.csv");
    
    // test corretness
    double MSE = 0;
    int n = model.f().rows();
    int m = time_mesh.rows();
    for (size_t i=0; i<n; ++i)
    {
        MSE += (sol(i) - model.f()(i))*(sol(i) - model.f()(i)) / (n);
    }
    std::cout << "MSE = " << MSE << std::endl;

    // EXPECT_TRUE(almost_equal(model.f(), sol));
    std::cout << (model.f()-sol).lpNorm<Eigen::Infinity>() << std::endl;
    EXPECT_TRUE(almost_equal(model.f(), sol));
}

// test 3 - method iterative
//    domain:       unit cube [0,1] x [0,1] x [0,1]
//    sampling:     locations = nodes
//    penalization: -laplacian + advection(b) + nonlinear reaction
//    covariates:   no
//    BC:           Dirichlet + Neumann
//    order FE:     1
//    time penalization: parabolic (iterative solver)
//    exact solution   : (x + y^2 + z^2)*t
TEST(strpde_nonlninear_test, cube_nonparametric_samplingatnodes_parabolic_iterative) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(10);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 1e-4*(i+1);
    // define spatial domain
    MeshLoader<Mesh3D> domain("unit_cube_5");
    // define the boundary with a DMatrix (=0 if Dirichlet, =1 if Neumann, =2 if Robin)
    // considering the unit cube,
    // we have Neumann boundary when z=0 and z=1 (upper and lower sides)
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1) ; // has all zeros
    for (size_t j=0; j<36; ++j) boundary_matrix(j, 0) = 1;
    for (size_t j=180; j<216; ++j) boundary_matrix(j, 0) = 1;
    // set the Dirichlet nodes at the edges
    for (size_t j=0; j<6; j++) {
        boundary_matrix(6*j, 0) = 0;      // lower face, points with x=0
        boundary_matrix(5 + j*6, 0) = 0; // lower face, points with x=1
        boundary_matrix(j, 0) = 0;         // lower face, points with y=0
        boundary_matrix(30 + j, 0) = 0;   // lower face, points with y=1

        boundary_matrix(180 + 6*j, 0) = 0; // upper face, points with x=0
        boundary_matrix(185 + 6*j, 0) = 0; // upper face, points with x=1
        boundary_matrix(180 + j, 0) = 0;    // upper face, points with y=0
        boundary_matrix(210 + j, 0) = 0;    // upper face, points with y=1
    }

    // import data from files
    DMatrix<double> y  = read_csv<double>("../data/models/strpde_nonlinear/3D_test3_cube5_samplingatnodes/y.csv");
    DMatrix<double> IC = read_csv<double>("../data/models/strpde_nonlinear/3D_test3_cube5_samplingatnodes/IC.csv");
    // define regularizing PDE
    SVector<3> b_; b_ << 2., -1., 1.;  // advection coefficient
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 1 - ff[0];};
    NonLinearReaction<3, LagrangianBasis<decltype(domain.mesh),1>::ReferenceBasis> non_linear_reaction(h_);
    auto L = dt<FEM>() -laplacian<FEM>() + advection<FEM>(b_) + non_linear_op<FEM>(non_linear_reaction);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, time_mesh, L, h_, boundary_matrix);
    pde.set_initial_condition(IC);
    //set forcing
    auto forcing_expr = [](SVector<3> x, double t) -> double {
        return x[0] + x[1]*x[1] + x[2]*x[2] - 4*t + (2 - 2*x[1] + 2*x[2])*t + (x[0] + x[1]*x[1] + x[2]*x[2])*t*(1 - (x[0] + x[1]*x[1] + x[2]*x[2])*t);
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);
    // set dirichlet bcs
    auto solution_expr = [](SVector<3> x, double t) -> double {
        return (x[0] + x[1]*x[1] + x[2]*x[2])*t;
    };
    DMatrix<double> nodes_ = pde.dof_coords();
    DMatrix<double> dirichlet_bc(nodes_.rows(), time_mesh.size());
    for (int i = 0; i < nodes_.rows(); ++i) {
        for (int j = 0; j < time_mesh.size(); ++j) {
            dirichlet_bc(i, j) = solution_expr(nodes_.row(i), time_mesh(j));
        }
    }
    pde.set_dirichlet_bc(dirichlet_bc);
    // set neumann bcs
    auto neumann_expr = [](SVector<3> x, double t) -> double { 
        return 2*x[2]*t;
    };
    DMatrix<double> boundary_quadrature_nodes = pde.boundary_quadrature_nodes();
    DMatrix<double> f_neumann(boundary_quadrature_nodes.rows(), time_mesh.size());
    for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
        for (int j = 0; j < time_mesh.size(); ++j) f_neumann(i, j) = neumann_expr(boundary_quadrature_nodes.row(i), time_mesh(j));
    }
    pde.set_neumann_bc(f_neumann);
    // define model
    double lambda_D = 1;
    double lambda_T = 1;
    STRPDE_NonLinear<SpaceTimeParabolic, fdapde::iterative> model(pde, Sampling::mesh_nodes);
    model.set_lambda_D(lambda_D);
    model.set_lambda_T(lambda_T);
    // set model's data
    BlockFrame<double, int> df;
    df.stack(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // set parameters for iterative method
    model.set_tolerance(1e-5);
    model.set_max_iter(50);
    // solve smoothing problem
    model.init();
    model.solve();
    DMatrix<double> sol = read_csv<double>("../data/models/strpde_nonlinear/3D_test3_cube5_samplingatnodes/sol.csv");
    
    // test corretness
    double MSE = 0;
    int n = model.f().rows();
    int m = time_mesh.rows();
    for (size_t i=0; i<n; ++i)
    {
        MSE += (sol(i) - model.f()(i))*(sol(i) - model.f()(i)) / (n);
    }
    std::cout << "MSE = " << MSE << std::endl;

    // EXPECT_TRUE(almost_equal(model.f(), sol));
    std::cout << (model.f()-sol).lpNorm<Eigen::Infinity>() << std::endl;
    EXPECT_TRUE(almost_equal(model.f(), sol));
}

// test 3 - method monolithic
//    domain:       unit cube [0,1] x [0,1] x [0,1]
//    sampling:     locations = nodes
//    penalization: -laplacian + advection(b) + nonlinear reaction
//    covariates:   no
//    BC:           Dirichlet + Neumann
//    order FE:     1
//    time penalization: parabolic (iterative solver)
//    exact solution   : (x + y^2 + z^2)*t
TEST(strpde_nonlninear_test, cube_nonparametric_samplingatnodes_parabolic_monolithic) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(10);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 1e-4*(i+1);
    // define spatial domain
    MeshLoader<Mesh3D> domain("unit_cube_5");
    // define the boundary with a DMatrix (=0 if Dirichlet, =1 if Neumann, =2 if Robin)
    // considering the unit cube,
    // we have Neumann boundary when z=0 and z=1 (upper and lower sides)
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1) ; // has all zeros
    for (size_t j=0; j<36; ++j) boundary_matrix(j, 0) = 1;
    for (size_t j=180; j<216; ++j) boundary_matrix(j, 0) = 1;
    // set the Dirichlet nodes at the edges
    for (size_t j=0; j<6; j++) {
        boundary_matrix(6*j, 0) = 0;      // lower face, points with x=0
        boundary_matrix(5 + j*6, 0) = 0; // lower face, points with x=1
        boundary_matrix(j, 0) = 0;         // lower face, points with y=0
        boundary_matrix(30 + j, 0) = 0;   // lower face, points with y=1

        boundary_matrix(180 + 6*j, 0) = 0; // upper face, points with x=0
        boundary_matrix(185 + 6*j, 0) = 0; // upper face, points with x=1
        boundary_matrix(180 + j, 0) = 0;    // upper face, points with y=0
        boundary_matrix(210 + j, 0) = 0;    // upper face, points with y=1
    }

    // import data from files
    DMatrix<double> y  = read_csv<double>("../data/models/strpde_nonlinear/3D_test3_cube5_samplingatnodes/y.csv");
    DMatrix<double> IC = read_csv<double>("../data/models/strpde_nonlinear/3D_test3_cube5_samplingatnodes/IC.csv");
    // define regularizing PDE
    SVector<3> b_; b_ << 2., -1., 1.;  // advection coefficient
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 1 - ff[0];};
    NonLinearReaction<3, LagrangianBasis<decltype(domain.mesh),1>::ReferenceBasis> non_linear_reaction(h_);
    auto L = dt<FEM>() -laplacian<FEM>() + advection<FEM>(b_) + non_linear_op<FEM>(non_linear_reaction);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, time_mesh, L, h_, boundary_matrix);
    pde.set_initial_condition(IC);
    //set forcing
    auto forcing_expr = [](SVector<3> x, double t) -> double {
        return x[0] + x[1]*x[1] + x[2]*x[2] - 4*t + (2 - 2*x[1] + 2*x[2])*t + (x[0] + x[1]*x[1] + x[2]*x[2])*t*(1 - (x[0] + x[1]*x[1] + x[2]*x[2])*t);
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);
    // set dirichlet bcs
    auto solution_expr = [](SVector<3> x, double t) -> double {
        return (x[0] + x[1]*x[1] + x[2]*x[2])*t;
    };
    DMatrix<double> nodes_ = pde.dof_coords();
    DMatrix<double> dirichlet_bc(nodes_.rows(), time_mesh.size());
    for (int i = 0; i < nodes_.rows(); ++i) {
        for (int j = 0; j < time_mesh.size(); ++j) {
            dirichlet_bc(i, j) = solution_expr(nodes_.row(i), time_mesh(j));
        }
    }
    pde.set_dirichlet_bc(dirichlet_bc);
    // set neumann bcs
    auto neumann_expr = [](SVector<3> x, double t) -> double { 
        return 2*x[2]*t;
    };
    DMatrix<double> boundary_quadrature_nodes = pde.boundary_quadrature_nodes();
    DMatrix<double> f_neumann(boundary_quadrature_nodes.rows(), time_mesh.size());
    for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
        for (int j = 0; j < time_mesh.size(); ++j) f_neumann(i, j) = neumann_expr(boundary_quadrature_nodes.row(i), time_mesh(j));
    }
    pde.set_neumann_bc(f_neumann);
    // define model
    double lambda_D = 1;
    double lambda_T = 1;
    STRPDE_NonLinear<SpaceTimeParabolic, fdapde::monolithic> model(pde, Sampling::mesh_nodes);
    model.set_lambda_D(lambda_D);
    model.set_lambda_T(lambda_T);
    // set model's data
    BlockFrame<double, int> df;
    df.stack(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // set parameters for iterative method
    model.set_tolerance(1e-5);
    model.set_max_iter(50);
    // solve smoothing problem
    model.init();
    model.solve();
    DMatrix<double> sol = read_csv<double>("../data/models/strpde_nonlinear/3D_test3_cube5_samplingatnodes/sol.csv");
    
    // test corretness
    double MSE = 0;
    int n = model.f().rows();
    int m = time_mesh.rows();
    for (size_t i=0; i<n; ++i)
    {
        MSE += (sol(i) - model.f()(i))*(sol(i) - model.f()(i)) / (n);
    }
    std::cout << "MSE = " << MSE << std::endl;

    // EXPECT_TRUE(almost_equal(model.f(), sol));
    std::cout << (model.f()-sol).lpNorm<Eigen::Infinity>() << std::endl;
    EXPECT_TRUE(almost_equal(model.f(), sol));
}

// test 4 - method iterative
//    domain:       surface
//    sampling:     locations = nodes
//    penalization: -laplacian + nonlinear reaction
//    covariates:   no
//    BC:           Dirichlet
//    order FE:     1
//    time penalization: parabolic (iterative solver)
//    exact solution   : (cos(pi*x) + cos(pi*y) + cos(pi*z))*exp(-t)
TEST(strpde_nonlninear_test, surface_nonparametric_samplingatnodes_parabolic_iterative) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(10);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 1e-4*(i+1);
    // define spatial domain
    MeshLoader<SurfaceMesh> domain("surface");
    // define the boundary with a DMatrix (=0 if Dirichlet, =1 if Neumann, =2 if Robin)
    // considering the unit cube,
    // we have Neumann boundary when z=0 and z=1 (upper and lower sides)
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1) ; // has all zeros

    // import data from files
    DMatrix<double> y  = read_csv<double>("../data/models/strpde_nonlinear/2.5D_test4_samplingatnodes/y.csv");
    DMatrix<double> IC = read_csv<double>("../data/models/strpde_nonlinear/2.5D_test4_samplingatnodes/IC.csv");
    // define regularizing PDE
    std::function<double(SVector<1>)> h_ = [](SVector<1> ff) -> double {return 1 - ff[0];};
    NonLinearReaction<2, LagrangianBasis<decltype(domain.mesh),1>::ReferenceBasis> non_linear_reaction(h_);
    auto L = dt<FEM>() -laplacian<FEM>() + non_linear_op<FEM>(non_linear_reaction);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, time_mesh, L, h_, boundary_matrix);
    pde.set_initial_condition(IC);
    // set dirichlet bcs
    auto solution_expr = [](SVector<3> x, double t) -> double {
        return (std::cos(pi*x[0]) + std::cos(pi*x[1] + std::cos(pi*x[2]))) * std::exp(-t);
    };
    DMatrix<double> nodes_ = pde.dof_coords();
    DMatrix<double> dirichlet_bc(nodes_.rows(), time_mesh.size());
    for (int i = 0; i < nodes_.rows(); ++i) {
        for (int j = 0; j < time_mesh.size(); ++j) {
            dirichlet_bc(i, j) = solution_expr(nodes_.row(i), time_mesh(j));
        }
    }
    pde.set_dirichlet_bc(dirichlet_bc);
    //set forcing
    auto forcing_expr = [&](SVector<3> x, double t) -> double {
        return solution_expr(x,t)*(pi*pi - solution_expr(x,t));
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);
    // define model
    double lambda_D = 1;
    double lambda_T = 1;
    STRPDE_NonLinear<SpaceTimeParabolic, fdapde::iterative> model(pde, Sampling::mesh_nodes);
    model.set_lambda_D(lambda_D);
    model.set_lambda_T(lambda_T);
    // set model's data
    BlockFrame<double, int> df;
    df.stack(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // set parameters for iterative method
    model.set_tolerance(1e-5);
    model.set_max_iter(50);
    // solve smoothing problem
    model.init();
    model.solve();
    DMatrix<double> sol = read_csv<double>("../data/models/strpde_nonlinear/2.5D_test4_samplingatnodes/sol.csv");
    
    // test corretness
    double MSE = 0;
    int n = model.f().rows();
    int m = time_mesh.rows();
    for (size_t i=0; i<n; ++i)
    {
        MSE += (sol(i) - model.f()(i))*(sol(i) - model.f()(i)) / (n);
    }
    std::cout << "MSE = " << MSE << std::endl;

    // EXPECT_TRUE(almost_equal(model.f(), sol));
    std::cout << (model.f()-sol).lpNorm<Eigen::Infinity>() << std::endl;
    EXPECT_TRUE(almost_equal(model.f(), sol));
}

// test 5 - method iterative
//    domain:       unit square [1,1] x [1,1]
//    sampling:     space locations != nodes
//    penalization: laplacian + nonlinear reaction
//    covariates:   yes
//    BC:           Dirichlet
//    order FE:     1
//    time penalization: parabolic (iterative solver)
//    exact solution   : f = (cos(pi*x)*cos(pi*y) + 2)*exp(-t); beta = [0.5, 3]
// TEST(strpde_nonlninear_test, laplacian_nonparametric_sapcesamplingpoitwise_covariates_parabolic_monolithic) {
//     // define temporal domain
//     DVector<double> time_mesh;
//     time_mesh.resize(10);
//     for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 1e-4*(i+1);
//     // define spatial domain
//     MeshLoader<Mesh2D> domain("unit_square_coarse");
//     DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1) ; // has all zeros
//     // import data from files
//     DMatrix<double> locs = read_csv<double>("../data/models/strpde_nonlinear/2D_test5/space_locs.csv");
//     DMatrix<double> y    = read_csv<double>("../data/models/strpde_nonlinear/2D_test5/y.csv");
//     DMatrix<double> X    = read_csv<double>("../data/models/strpde_nonlinear/2D_test5/X_samplings.csv");
//     DMatrix<double> IC   = read_csv<double>("../data/models/strpde_nonlinear/2D_test5/IC.csv");
//     // define regularizing PDE
//     std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 1 - ff[0];};
//     NonLinearReaction<2, LagrangianBasis<decltype(domain.mesh),1>::ReferenceBasis> non_linear_reaction(h_);
//     auto L = dt<FEM>() - laplacian<FEM>() - non_linear_op<FEM>(non_linear_reaction);
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, time_mesh, L, boundary_matrix);
//     pde.set_initial_condition(IC);
//     //set forcing
//     auto forcing_expr = [](SVector<2> x, double t) -> double {
//         return -4*std::exp(-t) + std::cos(pi*x[0])*std::cos(pi*x[1])*std::exp(-t)*(-2+2*pi*pi) + (4 + std::cos(pi*x[0])*std::cos(pi*x[0])*std::cos(pi*x[1])*std::cos(pi*x[1]) + 4*std::cos(pi*x[0])*std::cos(pi*x[1]))*std::exp(-2*t);
//     };
//     DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
//     DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
//     for (int i = 0; i < quadrature_nodes.rows(); ++i) {
//         for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
//     }
//     pde.set_forcing(u);
//     // set dirichlet bcs
//     auto solution_expr = [](SVector<2> x, double t) -> double {
//         return (std::cos(pi*x[0])*std::cos(pi*x[1]) + 2)*std::exp(-t);
//     };
//     DMatrix<double> nodes_ = pde.dof_coords();
//     DMatrix<double> dirichlet_bc(nodes_.rows(), time_mesh.size());
//     for (int i = 0; i < nodes_.rows(); ++i) {
//         for (int j = 0; j < time_mesh.size(); ++j) {
//             dirichlet_bc(i, j) = solution_expr(nodes_.row(i), time_mesh(j));
//         }
//     }
//     pde.set_dirichlet_bc(dirichlet_bc);
//     // define model
//     double lambda_D = 1;
//     double lambda_T = 1;
//     STRPDE_NonLinear<SpaceTimeParabolic, fdapde::monolithic> model(pde, Sampling::pointwise);
//     model.set_lambda_D(lambda_D);
//     model.set_lambda_T(lambda_T);
//     model.set_spatial_locations(locs);
//     // set model's data
//     BlockFrame<double, int> df;
//     df.stack(OBSERVATIONS_BLK, y);
//     std::cout << "debug" << std::endl;
//     df.stack(DESIGN_MATRIX_BLK, X);
//     std::cout << "debug" << std::endl;
//     model.set_data(df);
//     std::cout << "debug" << std::endl;
//     // set parameters for iterative method
//     model.set_tolerance(1e-5);
//     model.set_max_iter(50);
//     // solve smoothing problem
//     model.init();
//     std::cout << "debug" << std::endl;
//     model.solve();
//     std::cout << "debug" << std::endl;
//     DMatrix<double> sol = read_csv<double>("../data/models/strpde_nonlinear/2D_test5/sol.csv");
//     DMatrix<double> beta_exact = read_csv<double>("../data/models/strpde_nonlinear/2D_test5/beta.csv");
//
//     // test corretness
//     DVector<double> f_in_sampling_points(locs.rows(), 1);
//     int m = time_mesh.rows();
//     SpMatrix<double> Im(m, m);
//     Im.setIdentity();
//     f_in_sampling_points = Kronecker(Im, model.Psi())*model.f();
//     DMatrix<double> sol_in_sampling_points = read_csv<double>("../data/models/strpde_nonlinear/2D_test5/sol_samplingpoints.csv");
//     double MSE_sampling = 0;
//     int n = f_in_sampling_points.rows();
//     for (size_t i=0; i<n; ++i)
//     {
//         MSE_sampling += (sol_in_sampling_points(i) - f_in_sampling_points(i))*(sol_in_sampling_points(i) - f_in_sampling_points(i)) / (n);
//     }
//     std::cout << "MSE_sampling = " << MSE_sampling << std::endl;
//     std::cout << "inf_norm error in sampling points = " << (f_in_sampling_points-sol_in_sampling_points).lpNorm<Eigen::Infinity>() << std::endl;

//     double MSE = 0;
//     n = model.f().rows();
//     m = time_mesh.rows();
//     for (size_t i=0; i<n; ++i)
//     {
//         MSE += (sol(i) - model.f()(i))*(sol(i) - model.f()(i)) / (n);
//     }
//     std::cout << "MSE = " << MSE << std::endl;

//     std::cout << model.beta() << std::endl;
//     std::cout << beta_exact << std::endl;

//     // EXPECT_TRUE(almost_equal(model.f(), sol));
//     std::cout << (model.f()-sol).lpNorm<Eigen::Infinity>() << std::endl;
//     EXPECT_TRUE(almost_equal(model.f(), sol));
//     EXPECT_TRUE(almost_equal(model.beta(), beta_exact));
// }