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
using fdapde::core::reaction;
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

#include <chrono>
#include <thread>




// test 1 - method strpde monolithic
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: laplacian
//    covariates:   no
//    BC:           Dirichlet
//    order FE:     1
//    time penalization: parabolic (monolithic solver)
//    exact solution   : (cos(pi*x)*cos(pi*y) + 2)*exp(-t)
TEST(strpde_nonlninear_test2, laplacian_nonparametric_samplingatnodes_parabolic_strpde_monolithic) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(10);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = (0.1)*(i+1);
    // define spatial domain
    MeshLoader<Mesh2D> domain("unit_square_coarse");
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1);
    // // neumann boundary on the left and right side (x=0, x=1)
    // for (size_t j=1; j<20; j++) {
    //     boundary_matrix(0 + 21*j, 0) = 1;
    //     boundary_matrix(20 + 21*j, 0) = 1;
    // }

    // import data from files
    DMatrix<double> locs  = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_10/space_locs.csv");
    DMatrix<double> IC    = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_10/IC.csv");
    // define regularizing PDE
    auto L = dt<FEM>() - laplacian<FEM>();
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, time_mesh, L, boundary_matrix);
    pde.set_initial_condition(IC);
    //set forcing
    double alpha = 7;
    auto forcing_expr = [&](SVector<2> x, double t) -> double {
        return -((x[0]*x[0] + x[1]*x[1] + 2)*std::exp(-t)) - 4*std::exp(-t) + 20*((x[0]*x[0] + x[1]*x[1] + 2)*std::exp(-t))/(5 + ((x[0]*x[0] + x[1]*x[1] + 2)*std::exp(-t)));
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);
    // set dirichlet bcs
    auto solution_expr = [](SVector<2> x, double t) -> double {
        return (x[0]*x[0] + x[1]*x[1] + 2) * std::exp(-t);
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
    // auto neumann_expr = [](SVector<2> x, double t) -> double { 
    //     if (x[1]==0 || x[1]==1) return 0;
    //     return -pi*std::sin(pi*x[0])*std::cos(pi*x[1])*std::exp(-t);
    //     // return 2*x[0]*t;;
    // };
    // DMatrix<double> boundary_quadrature_nodes = pde.boundary_quadrature_nodes();
    // DMatrix<double> f_neumann(boundary_quadrature_nodes.rows(), time_mesh.size());
    // for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
    //     for (int j = 0; j < time_mesh.size(); ++j) f_neumann(i, j) = neumann_expr(boundary_quadrature_nodes.row(i), time_mesh(j));
    // }
    // pde.set_neumann_bc(f_neumann);

    // save the results in a file
    std::ofstream file("results10_STRPDE_monolithic_newlambda.txt");    //it will be exported in the current build directory

    // read parameters lambda from a file
    std::ifstream infile("lambdas10_STRPDE_monolithic.txt");
    if (!infile) {
        std::cerr << "Unable to open file lambdas.txt";
    }
    double value;
    std::vector<double> lambdas_D;
    while (infile >> value) {
        lambdas_D.push_back(value);
    }
    infile.close();

    int n_obs = 30;
    for (size_t i=1; i<=n_obs; ++i) {
        std::cout << "Iteration " << i << std::endl;
        // define model
        double lambda_D = lambdas_D[i-1];
        double lambda_T = 1;
        STRPDE<SpaceTimeParabolic, fdapde::monolithic> model(pde, Sampling::pointwise);
        model.set_lambda_D(lambda_D);
        model.set_lambda_T(lambda_T);
        model.set_spatial_locations(locs);
        // set model's data
        std::string filename = "../data/models/strpde_nonlinear/test nuovo/test_nuovo_10/y" + std::to_string(i) + ".csv";
        DMatrix<double> y = read_csv<double>(filename);
        BlockFrame<double, int> df;
        df.stack(OBSERVATIONS_BLK, y);
        model.set_data(df);
        // initialize smoothing problem
        model.init();
        double time = 0;
        auto start = std::chrono::high_resolution_clock::now();
        model.solve();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        time = duration.count();

        // compute MSE over the grid nodes
        // create a grid of points different from the sampling points where we can compute the MSE
        STRPDE<SpaceTimeParabolic, fdapde::iterative> model_grid(pde, Sampling::pointwise);
        DMatrix<double> grid = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/space_locs_grid.csv");
        model_grid.set_spatial_locations(grid);
        model_grid.init_sampling(true);  // evalute Psi() in the new grid
        // compute MSE over the grid
        DVector<double> f_grid(grid.rows(), 1);
        f_grid = model_grid.Psi()*model.f();
        DMatrix<double> sol_grid = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/sol_grid.csv");
        double MSE_grid = 0;
        int n = f_grid.rows();
        for (size_t i=0; i<n; ++i)
        {
            MSE_grid += (sol_grid(i) - f_grid(i))*(sol_grid(i) - f_grid(i)) / (n);
        }
        file << "\"" << i+150 << "\" " << 1000 << " \"method1\" " << time << " " << MSE_grid << "\n";
        // file << model.f();
    }

    file.close();
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
TEST(strpde_nonlninear_test2, laplacian_nonparametric_samplingatnodes_parabolic_strpde_iterative) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(10);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = (0.1)*(i+1);
    // define spatial domain
    MeshLoader<Mesh2D> domain("unit_square_coarse");
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1);
    // // neumann boundary on the left and right side (x=0, x=1)
    // for (size_t j=1; j<20; j++) {
    //     boundary_matrix(0 + 21*j, 0) = 1;
    //     boundary_matrix(20 + 21*j, 0) = 1;
    // }

    // import data from files
    DMatrix<double> locs  = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_10/space_locs.csv");
    DMatrix<double> IC    = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_10/IC.csv");
    // define regularizing PDE
    auto L = dt<FEM>() - laplacian<FEM>();
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, time_mesh, L, boundary_matrix);
    pde.set_initial_condition(IC);
    //set forcing
    double alpha = 7;
    auto forcing_expr = [&](SVector<2> x, double t) -> double {
        return -((x[0]*x[0] + x[1]*x[1] + 2) * std::exp(-t)) - 4*std::exp(-t) + 20*((x[0]*x[0] + x[1]*x[1] + 2) * std::exp(-t))/(5 + ((x[0]*x[0] + x[1]*x[1] + 2) * std::exp(-t)));
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);
    // set dirichlet bcs
    auto solution_expr = [](SVector<2> x, double t) -> double {
        return (x[0]*x[0] + x[1]*x[1] + 2) * std::exp(-t);
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
    // auto neumann_expr = [](SVector<2> x, double t) -> double { 
    //     if (x[1]==0 || x[1]==1) return 0;
    //     return -pi*std::sin(pi*x[0])*std::cos(pi*x[1])*std::exp(-t);
    //     // return 2*x[0]*t;;
    // };
    // DMatrix<double> boundary_quadrature_nodes = pde.boundary_quadrature_nodes();
    // DMatrix<double> f_neumann(boundary_quadrature_nodes.rows(), time_mesh.size());
    // for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
    //     for (int j = 0; j < time_mesh.size(); ++j) f_neumann(i, j) = neumann_expr(boundary_quadrature_nodes.row(i), time_mesh(j));
    // }
    // pde.set_neumann_bc(f_neumann);

    // save the results in a file
    std::ofstream file("results10_STRPDE_iterative_newlambda.txt");    //it will be exported in the current build directory

    // read parameters lambda from a file
    std::ifstream infile("lambdas10_STRPDE_iterative.txt");
    if (!infile) {
        std::cerr << "Unable to open file lambdas.txt";
    }
    double value;
    std::vector<double> lambdas_D;
    while (infile >> value) {
        lambdas_D.push_back(value);
    }
    infile.close();

    int n_obs = 30;
    for (size_t i=1; i<=n_obs; ++i) {
        std::cout << "Iteration " << i << std::endl;
        // define model
        double lambda_D = lambdas_D[i-1];
        double lambda_T = 1;
        STRPDE<SpaceTimeParabolic, fdapde::iterative> model(pde, Sampling::pointwise);
        model.set_lambda_D(lambda_D);
        model.set_lambda_T(lambda_T);
        model.set_spatial_locations(locs);
        // set model's data
        std::string filename = "../data/models/strpde_nonlinear/test nuovo/test_nuovo_10/y" + std::to_string(i) + ".csv";
        DMatrix<double> y = read_csv<double>(filename);
        BlockFrame<double, int> df;
        df.stack(OBSERVATIONS_BLK, y);
        model.set_data(df);
        // set parameters for iterative method
        model.set_tolerance(1e-5);
        model.set_max_iter(50);
        // initialize smoothing problem
        model.init();
        double time = 0;
        auto start = std::chrono::high_resolution_clock::now();
        model.solve();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        time = duration.count();

        // compute MSE over the grid nodes
        // create a grid of points different from the sampling points where we can compute the MSE
        STRPDE<SpaceTimeParabolic, fdapde::iterative> model_grid(pde, Sampling::pointwise);
        DMatrix<double> grid = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/space_locs_grid.csv");
        model_grid.set_spatial_locations(grid);
        model_grid.init_sampling(true);  // evalute Psi() in the new grid
        // compute MSE over the grid
        DVector<double> f_grid(grid.rows(), 1);
        f_grid = model_grid.Psi()*model.f();
        DMatrix<double> sol_grid = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/sol_grid.csv");
        double MSE_grid = 0;
        int n = f_grid.rows();
        for (size_t i=0; i<n; ++i)
        {
            MSE_grid += (sol_grid(i) - f_grid(i))*(sol_grid(i) - f_grid(i)) / (n);
        }
        file << "\"" << i+30+150 << "\" " << 1000 << " \"method2\" " << time << " " << MSE_grid << "\n";
        // file << model.f();
    }

    file.close();
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
TEST(strpde_nonlninear_test2, laplacian_nonparametric_samplingatnodes_parabolic_iterative_EI) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(10);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = (0.1)*(i+1);
    // define spatial domain
    MeshLoader<Mesh2D> domain("unit_square_coarse");
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1);
    // neumann boundary on the left and right side (x=0, x=1)
    // for (size_t j=1; j<20; j++) {
    //     boundary_matrix(0 + 21*j, 0) = 1;
    //     boundary_matrix(20 + 21*j, 0) = 1;
    // }
    // import data from files
    DMatrix<double> locs  = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_10/space_locs.csv");
    DMatrix<double> IC    = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_10/IC.csv");
    // define regularizing PDE
    double alpha = 7;
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 20/(5 + ff[0]);};
    NonLinearReaction<2, LagrangianBasis<decltype(domain.mesh),1>::ReferenceBasis> non_linear_reaction(h_);
    auto L = dt<FEM>() - laplacian<FEM>() + non_linear_op<FEM>(non_linear_reaction);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, time_mesh, L, h_, boundary_matrix);
    pde.set_initial_condition(IC);
    //set forcing
    auto forcing_expr = [&](SVector<2> x, double t) -> double {
        return -((x[0]*x[0] + x[1]*x[1] + 2)*std::exp(-t)) - 4*std::exp(-t) + 20*((x[0]*x[0] + x[1]*x[1] + 2)*std::exp(-t))/(5 + ((x[0]*x[0] + x[1]*x[1] + 2)*std::exp(-t)));
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);
    // set dirichlet bcs
    auto solution_expr = [](SVector<2> x, double t) -> double {
        return (x[0]*x[0] + x[1]*x[1] + 2) * std::exp(-t);
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
    // auto neumann_expr = [](SVector<2> x, double t) -> double { 
    //     if (x[1]==0 || x[1]==1) return 0;
    //     return -pi*std::sin(pi*x[0])*std::cos(pi*x[1])*std::exp(-t);
    //     // return 2*x[0]*t;;
    // };
    // DMatrix<double> boundary_quadrature_nodes = pde.boundary_quadrature_nodes();
    // DMatrix<double> f_neumann(boundary_quadrature_nodes.rows(), time_mesh.size());
    // for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
    //     for (int j = 0; j < time_mesh.size(); ++j) f_neumann(i, j) = neumann_expr(boundary_quadrature_nodes.row(i), time_mesh(j));
    // }
    // pde.set_neumann_bc(f_neumann);

    std::ofstream file("results10_NLSTRPDE_iterativeEI_newlambda.txt");    //it will be exported in the current build directory

    // read parameters lambda from a file
    std::ifstream infile("lambdas10_NLSTRPDE_iterativeEI.txt");
    if (!infile) {
        std::cerr << "Unable to open file lambdas.txt";
    }
    double value;
    std::vector<double> lambdas_D;
    while (infile >> value) {
        lambdas_D.push_back(value);
    }
    infile.close();

    int n_obs = 30;
    for (size_t i=1; i<=n_obs; ++i) {
        // std::cout << "Iteration " << i << std::endl;
        // define model
        double lambda_D = lambdas_D[i-1];
        double lambda_T = 1;
        STRPDE_NonLinear<SpaceTimeParabolic, fdapde::iterative_EI> model(pde, Sampling::pointwise);
        model.set_lambda_D(lambda_D);
        model.set_lambda_T(lambda_T);
        model.set_spatial_locations(locs);
        // set model's data
        std::string filename = "../data/models/strpde_nonlinear/test nuovo/test_nuovo_10/y" + std::to_string(i) + ".csv";
        DMatrix<double> y = read_csv<double>(filename);
        BlockFrame<double, int> df;
        df.stack(OBSERVATIONS_BLK, y);
        model.set_data(df);
        // set parameters for iterative method
        model.set_tolerance(1e-5, 1e-8);
        model.set_max_iter(50);
        // initialize smoothing problem
        model.init();
        double time = 0;
        auto start = std::chrono::high_resolution_clock::now();
        model.solve();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        time = duration.count();

        // compute MSE over the grid nodes
        // create a grid of points different from the sampling points where we can compute the MSE
        STRPDE<SpaceTimeParabolic, fdapde::iterative> model_grid(pde, Sampling::pointwise);
        DMatrix<double> grid = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/space_locs_grid.csv");
        model_grid.set_spatial_locations(grid);
        model_grid.init_sampling(true);  // evalute Psi() in the new grid
        // compute MSE over the grid
        DVector<double> f_grid(grid.rows(), 1);
        f_grid = model_grid.Psi()*model.f();
        DMatrix<double> sol_grid = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/sol_grid.csv");
        double MSE_grid = 0;
        int n = f_grid.rows();
        for (size_t i=0; i<n; ++i)
        {
            MSE_grid += (sol_grid(i) - f_grid(i))*(sol_grid(i) - f_grid(i)) / (n);
        }
        file << "\"" << i+120+150 << "\" " << 1000 << " \"method5\" " << time << " " << MSE_grid << "\n";
        // file << model.f();
    }

    file.close();
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
TEST(strpde_nonlninear_test2, laplacian_nonparametric_samplingatnodes_parabolic_iterative) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(10);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = (0.1)*(i+1);
    // define spatial domain
    MeshLoader<Mesh2D> domain("unit_square_coarse");
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1);
    // neumann boundary on the left and right side (x=0, x=1)
    // for (size_t j=1; j<20; j++) {
    //     boundary_matrix(0 + 21*j, 0) = 1;
    //     boundary_matrix(20 + 21*j, 0) = 1;
    // }
    // import data from files
    DMatrix<double> locs  = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_10/space_locs.csv");
    DMatrix<double> IC    = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_10/IC.csv");
    // define regularizing PDE
    double alpha = 7;
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 20/(5 + ff[0]);};
    NonLinearReaction<2, LagrangianBasis<decltype(domain.mesh),1>::ReferenceBasis> non_linear_reaction(h_);
    auto L = dt<FEM>() - laplacian<FEM>() + non_linear_op<FEM>(non_linear_reaction);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, time_mesh, L, h_, boundary_matrix);
    pde.set_initial_condition(IC);
    //set forcing
    auto forcing_expr = [&](SVector<2> x, double t) -> double {
        return -((x[0]*x[0] + x[1]*x[1] + 2) * std::exp(-t)) - 4*std::exp(-t) + 20*((x[0]*x[0] + x[1]*x[1] + 2) * std::exp(-t))/(5 + ((x[0]*x[0] + x[1]*x[1] + 2) * std::exp(-t)));
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);
    // set dirichlet bcs
    auto solution_expr = [](SVector<2> x, double t) -> double {
        return (x[0]*x[0] + x[1]*x[1] + 2) * std::exp(-t);
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
    // auto neumann_expr = [](SVector<2> x, double t) -> double { 
    //     if (x[1]==0 || x[1]==1) return 0;
    //     return -pi*std::sin(pi*x[0])*std::cos(pi*x[1])*std::exp(-t);
    //     // return 2*x[0]*t;;
    // };
    // DMatrix<double> boundary_quadrature_nodes = pde.boundary_quadrature_nodes();
    // DMatrix<double> f_neumann(boundary_quadrature_nodes.rows(), time_mesh.size());
    // for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
    //     for (int j = 0; j < time_mesh.size(); ++j) f_neumann(i, j) = neumann_expr(boundary_quadrature_nodes.row(i), time_mesh(j));
    // }
    // pde.set_neumann_bc(f_neumann);

    std::ofstream file("results10_NLSTRPDE_iterative_newlambda.txt");    //it will be exported in the current build directory

    // read parameters lambda from a file
    std::ifstream infile("lambdas10_NLSTRPDE_iterative.txt");
    if (!infile) {
        std::cerr << "Unable to open file lambdas.txt";
    }
    double value;
    std::vector<double> lambdas_D;
    while (infile >> value) {
        lambdas_D.push_back(value);
    }
    infile.close();

    int n_obs = 30;
    for (size_t i=1; i<=n_obs; ++i) {
        // std::cout << "Iteration " << i << std::endl;
        // define model
        double lambda_D = lambdas_D[i-1];
        double lambda_T = 1;
        STRPDE_NonLinear<SpaceTimeParabolic, fdapde::iterative> model(pde, Sampling::pointwise);
        model.set_lambda_D(lambda_D);
        model.set_lambda_T(lambda_T);
        model.set_spatial_locations(locs);
        // set model's data
        std::string filename = "../data/models/strpde_nonlinear/test nuovo/test_nuovo_10/y" + std::to_string(i) + ".csv";
        DMatrix<double> y = read_csv<double>(filename);
        BlockFrame<double, int> df;
        df.stack(OBSERVATIONS_BLK, y);
        model.set_data(df);
        // set parameters for iterative method
        model.set_tolerance(1e-5, 1e-8);
        model.set_max_iter(50);
        // initialize smoothing problem
        model.init();
        double time = 0;
        auto start = std::chrono::high_resolution_clock::now();
        model.solve();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        time = duration.count();

        // compute MSE over the grid nodes
        // create a grid of points different from the sampling points where we can compute the MSE
        STRPDE<SpaceTimeParabolic, fdapde::iterative> model_grid(pde, Sampling::pointwise);
        DMatrix<double> grid = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/space_locs_grid.csv");
        model_grid.set_spatial_locations(grid);
        model_grid.init_sampling(true);  // evalute Psi() in the new grid
        // compute MSE over the grid
        DVector<double> f_grid(grid.rows(), 1);
        f_grid = model_grid.Psi()*model.f();
        DMatrix<double> sol_grid = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/sol_grid.csv");
        double MSE_grid = 0;
        int n = f_grid.rows();
        for (size_t i=0; i<n; ++i)
        {
            MSE_grid += (sol_grid(i) - f_grid(i))*(sol_grid(i) - f_grid(i)) / (n);
        }
        file << "\"" << i+90+150 << "\" " << 1000 << " \"method4\" " << time << " " << MSE_grid << "\n";
        // file << model.f();
    }

    file.close();
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
TEST(strpde_nonlninear_test2, laplacian_nonparametric_samplingatnodes_parabolic_monolithic) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(10);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = (0.1)*(i+1);
    // define spatial domain
    MeshLoader<Mesh2D> domain("unit_square_coarse");
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1);
    // neumann boundary on the left and right side (x=0, x=1)
    // for (size_t j=1; j<20; j++) {
    //     boundary_matrix(0 + 21*j, 0) = 1;
    //     boundary_matrix(20 + 21*j, 0) = 1;
    // }
    // import data from files
    DMatrix<double> locs  = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_10/space_locs.csv");
    DMatrix<double> IC    = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_10/IC.csv");
    // define regularizing PDE
    double alpha = 7;
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 20/(5 + ff[0]);};
    NonLinearReaction<2, LagrangianBasis<decltype(domain.mesh),1>::ReferenceBasis> non_linear_reaction(h_);
    auto L = dt<FEM>() - laplacian<FEM>() + non_linear_op<FEM>(non_linear_reaction);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, time_mesh, L, h_, boundary_matrix);
    pde.set_initial_condition(IC);
    //set forcing
    auto forcing_expr = [&](SVector<2> x, double t) -> double {
        return -((x[0]*x[0] + x[1]*x[1] + 2) * std::exp(-t)) - 4*std::exp(-t) + 20*((x[0]*x[0] + x[1]*x[1] + 2) * std::exp(-t))/(5 + ((x[0]*x[0] + x[1]*x[1] + 2) * std::exp(-t)));
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);
    // set dirichlet bcs
    auto solution_expr = [](SVector<2> x, double t) -> double {
        return (x[0]*x[0] + x[1]*x[1] + 2) * std::exp(-t);
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
    // auto neumann_expr = [](SVector<2> x, double t) -> double { 
    //     if (x[1]==0 || x[1]==1) return 0;
    //     return -pi*std::sin(pi*x[0])*std::cos(pi*x[1])*std::exp(-t);
    //     // return 2*x[0]*t;;
    // };
    // DMatrix<double> boundary_quadrature_nodes = pde.boundary_quadrature_nodes();
    // DMatrix<double> f_neumann(boundary_quadrature_nodes.rows(), time_mesh.size());
    // for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
    //     for (int j = 0; j < time_mesh.size(); ++j) f_neumann(i, j) = neumann_expr(boundary_quadrature_nodes.row(i), time_mesh(j));
    // }
    // pde.set_neumann_bc(f_neumann);

    std::ofstream file("results10_NLSTRPDE_monolithic_newlambda.txt");    //it will be exported in the current build directory

    // read parameters lambda from a file
    std::ifstream infile("lambdas10_NLSTRPDE_monolithic.txt");
    if (!infile) {
        std::cerr << "Unable to open file lambdas.txt";
    }
    double value;
    std::vector<double> lambdas_D;
    while (infile >> value) {
        lambdas_D.push_back(value);
    }
    infile.close();

    int n_obs = 30;
    for (size_t i=1; i<=n_obs; ++i) {
        // std::cout << "Iteration " << i << std::endl;
        // define model
        double lambda_D = lambdas_D[i-1];
        double lambda_T = 1;
        STRPDE_NonLinear<SpaceTimeParabolic, fdapde::monolithic> model(pde, Sampling::pointwise);
        model.set_lambda_D(lambda_D);
        model.set_lambda_T(lambda_T);
        model.set_spatial_locations(locs);
        // set model's data
        std::string filename = "../data/models/strpde_nonlinear/test nuovo/test_nuovo_10/y" + std::to_string(i) + ".csv";
        DMatrix<double> y = read_csv<double>(filename);
        BlockFrame<double, int> df;
        df.stack(OBSERVATIONS_BLK, y);
        model.set_data(df);
        // set parameters for iterative method
        model.set_tolerance(1e-5);
        model.set_max_iter(15);
        // initialize smoothing problem
        model.init();
        double time = 0;
        auto start = std::chrono::high_resolution_clock::now();
        model.solve();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        time = duration.count();

        // compute MSE over the grid nodes
        // create a grid of points different from the sampling points where we can compute the MSE
        STRPDE<SpaceTimeParabolic, fdapde::iterative> model_grid(pde, Sampling::pointwise);
        DMatrix<double> grid = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/space_locs_grid.csv");
        model_grid.set_spatial_locations(grid);
        model_grid.init_sampling(true);  // evalute Psi() in the new grid
        // compute MSE over the grid
        DVector<double> f_grid(grid.rows(), 1);
        f_grid = model_grid.Psi()*model.f();
        DMatrix<double> sol_grid = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/sol_grid.csv");
        double MSE_grid = 0;
        int n = f_grid.rows();
        for (size_t i=0; i<n; ++i)
        {
            MSE_grid += (sol_grid(i) - f_grid(i))*(sol_grid(i) - f_grid(i)) / (n);
        }
        file << "\"" << i+60+150 << "\" " << 1000 << " \"method3\" " << time << " " << MSE_grid << "\n";
        // file << model.f();
    }

    file.close();
}