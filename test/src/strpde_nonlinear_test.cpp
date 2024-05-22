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


/* TEST(strpde_nonlinear_test, nonparametric_samplingatnodes_parabolic_monolithic) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(10);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 1e-5 * i;
    // define spatial domain
    MeshLoader<Mesh2D> domain("unit_square_coarse");
    // import data from files
    DMatrix<double> y  = read_mtx<double>("../data/models/strpde/2D_test4/y.mtx" );
    DMatrix<double> IC = read_mtx<double>("../data/models/strpde/2D_test4/IC.mtx");   // initial condition
    // std::ofstream file("IC.txt");    //it will be exported in the current build directory
    // if (file.is_open()){
    //     for(int i = 0; i < IC.rows(); ++i)
    //         file << IC.col(0)(i) << '\n';
    //     file.close();
    // } else {
    //     std::cerr << "unable to save" << std::endl;
    // }
    // define regularizing PDE
    // define the nonlinearity: non linear reaction h_(u)*u
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 1 - ff[0];};
    NonLinearReaction<2, LagrangianBasis<decltype(domain.mesh),1>::ReferenceBasis> non_linear_reaction(h_);
    auto L =  dt<FEM>() - laplacian<FEM>() - non_linear_op<FEM>(non_linear_reaction);
    // define the PDE
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, time_mesh, L, h_);
    // define the force
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 6, time_mesh.rows());
    auto forcing_expr = [](SVector<2> x, double t) -> double {
        return -4*std::exp(-t) + std::cos(pi*x[0])*std::cos(pi*x[1])*std::exp(-t)*(-2+2*pi*pi) + (4 + std::cos(pi*x[0])*std::cos(pi*x[0])*std::cos(pi*x[1])*std::cos(pi*x[1]) + 4*std::cos(pi*x[0])*std::cos(pi*x[1]))*std::exp(-2*t);
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);
    // set initial condition
    std::cout << "debug 1" << std::endl;
    pde.set_initial_condition(IC);
    // define model
    double lambda_D = std::pow(0.1, 6);
    double lambda_T = std::pow(0.1, 6);
    std::cout << "debug 2" << std::endl;
    STRPDE_NonLinear<SpaceTimeParabolic, fdapde::monolithic> model(pde, Sampling::mesh_nodes);
    std::cout << "debug 3" << std::endl;
    model.set_lambda_D(lambda_D);
    model.set_lambda_T(lambda_T);
    // set model's data
    BlockFrame<double, int> df;
    df.stack(OBSERVATIONS_BLK, y);
    std::cout << "debug 4" << std::endl;
    model.set_data(df);
    // solve smoothing problem
    std::cout << "debug 5" << std::endl;
    model.init();
    std::cout << "debug 6" << std::endl;
    model.solve();
    // test correctness
    EXPECT_TRUE(almost_equal(model.f(), "../data/models/strpde/2D_test4/sol.mtx"));
} */


// test 4
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   no
//    BC:           no
//    order FE:     1
//    time penalization: parabolic (iterative solver)
//    exact solution   : (cos(pi*x)*cos(pi*y) + 2)*exp(-t)
TEST(strpde_nonlninear_test, laplacian_nonparametric_samplingatnodes_parabolic_iterative) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(10);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 1e-4*(i+1);
    // define spatial domain
    MeshLoader<Mesh2D> domain("unit_square");
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1) ; // has all zeros
    // import data from files
    DMatrix<double> y  = read_csv<double>("../data/models/strpde_nonlinear/2D_test1/y.csv");
    DMatrix<double> IC = read_csv<double>("../data/models/strpde_nonlinear/2D_test1/IC.csv");
    // define regularizing PDE
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 1 - ff[0];};
    NonLinearReaction<2, LagrangianBasis<decltype(domain.mesh),1>::ReferenceBasis> non_linear_reaction(h_);
    auto L = dt<FEM>() - laplacian<FEM>() - non_linear_op<FEM>(non_linear_reaction);
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
    DMatrix<double> sol = read_csv<double>("../data/models/strpde_nonlinear/2D_test1/sol.csv");
    // test corretness
    std::ofstream file("f50.txt");    //it will be exported in the current build directory
    if (file.is_open()){
        for(int i = 0; i < sol.rows(); ++i)
            file << model.f().col(0)(i) << '\n';
        file.close();
    } else {
        std::cerr << "unable to save" << std::endl;
    }
    DMatrix<double> error_L2(time_mesh.rows(), 1);
    DMatrix<double> error_(nodes_.rows(), 1);
    for (int j = 0; j < time_mesh.rows(); ++j) {
        for (int i=0; i<nodes_.rows(); ++i) {
            error_(i) = sol.col(0)(i + j*nodes_.rows()) - model.f().col(0)(i + j*nodes_.rows());
        }
        auto eee = error_.cwiseProduct(error_);
        // std::cout << eee << std::endl;
        // std::cout << "error size = " << eee.rows() << "x" << eee.cols() << std::endl;
        // std::cout << "mass size = " << model.R0().rows() << "x" << model.R0().cols() << std::endl;
        error_L2(j, 0) = (model.R0() * error_.cwiseProduct(error_)).sum();
        std::cout << "t = " << j << " ErrorL2 = " << std::sqrt(error_L2(j,0)) << std::endl;
    }

    double MSE = 0;
    int n = model.f().rows();
    int m = time_mesh.rows();
    for (size_t i=0; i<n; ++i)
    {
        MSE += (sol(i) - model.f()(i))*(sol(i) - model.f()(i)) / (m*n);
    }
    std::cout << "MSE = " << MSE << std::endl;

    // EXPECT_TRUE(almost_equal(model.f(), sol));
    std::cout << (model.f()-sol).lpNorm<Eigen::Infinity>() << std::endl;
    EXPECT_TRUE(almost_equal(model.f(), sol));
}


// test 4
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   no
//    BC:           no
//    order FE:     1
//    time penalization: parabolic (iterative solver)
//    exact solution   : (cos(pi*x)*cos(pi*y) + 2)*exp(-t)
TEST(strpde_nonlninear_test, laplacian_nonparametric_samplingatnodes_parabolic_monolithic) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(10);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 1e-4*(i+1);
    // define spatial domain
    MeshLoader<Mesh2D> domain("unit_square");
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1);
    // import data from files
    DMatrix<double> y  = read_csv<double>("../data/models/strpde_nonlinear/2D_test1/y.csv");
    DMatrix<double> IC = read_csv<double>("../data/models/strpde_nonlinear/2D_test1/IC.csv");
    // define regularizing PDE
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 1 - ff[0];};
    NonLinearReaction<2, LagrangianBasis<decltype(domain.mesh),1>::ReferenceBasis> non_linear_reaction(h_);
    auto L = dt<FEM>() - laplacian<FEM>() - non_linear_op<FEM>(non_linear_reaction);
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
    std::cout << "debug 1" << std::endl;
    DMatrix<double> sol = read_csv<double>("../data/models/strpde_nonlinear/2D_test1/sol.csv");
    std::cout << "debug 2" << std::endl;
    // test corretness
    std::ofstream file("f_monolithic.txt");    //it will be exported in the current build directory
    if (file.is_open()){
        for(int i = 0; i < sol.rows(); ++i)
            file << model.f().col(0)(i) << '\n';
        file.close();
    } else {
        std::cerr << "unable to save" << std::endl;
    }
    std::cout << "debug 3" << std::endl;
    /* DMatrix<double> error_L2(time_mesh.rows(), 1);
    DMatrix<double> error_(nodes_.rows(), 1);
    for (int j = 0; j < time_mesh.rows(); ++j) {
        for (int i=0; i<nodes_.rows(); ++i) {
            error_(i) = sol.col(0)(i + j*nodes_.rows()) - model.f().col(0)(i + j*nodes_.rows());
        }
        auto eee = error_.cwiseProduct(error_);
        std::cout << "error size = " << eee.rows() << "x" << eee.cols() << std::endl;
        std::cout << "mass size = " << pde.mass().rows() << "x" << model.R0().cols() << std::endl;
        error_L2(j, 0) = (pde.mass() * error_.cwiseProduct(error_)).sum();
        std::cout << "t = " << j << " ErrorL2 = " << std::sqrt(error_L2(j,0)) << std::endl;
    } */
    std::cout << "debug 4" << std::endl;

    double MSE = 0;
    int n = model.f().rows();
    int m = time_mesh.rows();
    for (size_t i=0; i<n; ++i)
    {
        MSE += (sol(i) - model.f()(i))*(sol(i) - model.f()(i)) / (m*n);
    }
    std::cout << "MSE = " << MSE << std::endl;
    std::cout << "debug 5" << std::endl;

    // EXPECT_TRUE(almost_equal(model.f(), sol));
    std::cout << (model.f()-sol).lpNorm<Eigen::Infinity>() << std::endl;
    std::cout << "debug 6" << std::endl;
    EXPECT_TRUE(almost_equal(model.f(), sol));
}