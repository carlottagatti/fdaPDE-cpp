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

#include <fdaPDE/core.h>
#include <gtest/gtest.h>   // testing framework

#include <cstddef>
using fdapde::core::fem_order;
using fdapde::core::FEM;
using fdapde::core::laplacian;
using fdapde::core::PDE;

#include "../../fdaPDE/models/regression/srpde.h"
#include "../../fdaPDE/models/regression/qsrpde.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::SRPDE;
using fdapde::models::QSRPDE;
using fdapde::models::SpaceOnly;
using fdapde::models::Sampling;
#include "../../fdaPDE/calibration/kfold_cv.h"
#include "../../fdaPDE/calibration/rmse.h"
using fdapde::calibration::KCV;
using fdapde::calibration::RMSE;

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_mtx;
using fdapde::testing::read_csv;


// test 1 - method iterative
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: laplacian + nonlinear reaction
//    covariates:   no
//    BC:           Dirichlet
//    order FE:     1
//    time penalization: parabolic (iterative solver)
//    exact solution   : (cos(pi*x)*cos(pi*y) + 2)*exp(-t)
TEST(kcv_strpde_test_error, strpde_NLiterative_5error) {
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
    DMatrix<double> locs  = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_5/space_locs.csv");
    DMatrix<double> IC    = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_5/IC.csv");
    // define regularizing PDE
    double alpha = 7;
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 15/(5 + ff[0]);};
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

    std::ofstream file("lambdas5_NLSTRPDE_wrongPDE_iterative.txt");    //it will be exported in the current build directory

    int n_obs = 30;
    for (size_t i=1; i<=n_obs; ++i) {
        std::cout << "Iteration " << i << std::endl;
        // define model
        STRPDE_NonLinear<SpaceTimeParabolic, fdapde::iterative> model(pde, Sampling::pointwise);
        model.set_spatial_locations(locs);
        // set model's data
        std::string filename = "../data/models/strpde_nonlinear/test nuovo/test_nuovo_5/y" + std::to_string(i) + ".csv";
        DMatrix<double> y = read_csv<double>(filename);
        BlockFrame<double, int> df;
        df.stack(OBSERVATIONS_BLK, y);
        model.set_data(df);
        // set parameters for iterative method
        model.set_tolerance(1e-5, 1e-8);
        model.set_max_iter(50);
        // initialize smoothing problem
        model.init();

        // define KCV engine and search for best lambda which minimizes the model's RMSE
        std::size_t n_folds = 5;
        int seed = 435642;
        KCV kcv(n_folds, seed, time_mesh.size(), false);
        std::vector<DVector<double>> lambdas;
        for (double x = -5; x <= 0; x += 0.25) {
            lambdas.push_back(SVector<2>(std::pow(10, x), 1.0));
        }
        kcv.fit(model, lambdas, RMSE(model));

        std::cout << kcv.optimum() << std::endl;
        // std::cout << kcv.avg_scores() << std::endl;
        file << kcv.optimum()[0] << '\n';

        // define MinRMSE
        // MinRMSE min_rmse;
        // std::vector<DVector<double>> lambdas;
        // for (double x = -2.0; x >= -4; x -= 0.25) {
        //     lambdas.push_back(SVector<2>(std::pow(10, x), 1.0));
        // }
        // std::cout << "debug" << std::endl;
        // min_rmse.fit(model, lambdas);
        // std::cout << "debug" << std::endl;

        // std::cout << min_rmse.optimum() << std::endl;
        // std::cout << min_rmse.scores() << std::endl;
        //file << min_rmse.optimum()[0] << '\n';
    }

    file.close();
}

TEST(kcv_strpde_test_error, NLstrpde_iterativeEI_5error) {
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
    DMatrix<double> locs  = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_5/space_locs.csv");
    DMatrix<double> IC    = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_5/IC.csv");
    // define regularizing PDE
    double alpha = 7;
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 15/(5 + ff[0]);};
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

    std::ofstream file("lambdas5_NLSTRPDE_wrongPDE_iterative_EI.txt");    //it will be exported in the current build directory

    int n_obs = 30;
    for (size_t i=1; i<=n_obs; ++i) {
        std::cout << "Iteration " << i << std::endl;
        // define model
        STRPDE_NonLinear<SpaceTimeParabolic, fdapde::iterative_EI> model(pde, Sampling::pointwise);
        model.set_spatial_locations(locs);
        // set model's data
        std::string filename = "../data/models/strpde_nonlinear/test nuovo/test_nuovo_5/y" + std::to_string(i) + ".csv";
        DMatrix<double> y = read_csv<double>(filename);
        BlockFrame<double, int> df;
        df.stack(OBSERVATIONS_BLK, y);
        model.set_data(df);
        // set parameters for iterative method
        model.set_tolerance(1e-5, 1e-8);
        model.set_max_iter(50);
        // initialize smoothing problem
        model.init();

        // define KCV engine and search for best lambda which minimizes the model's RMSE
        std::size_t n_folds = 5;
        int seed = 435642;
        KCV kcv(n_folds, seed, time_mesh.size(), false);
        std::vector<DVector<double>> lambdas;
        for (double x = -5; x <= 0; x += 0.25) {
            lambdas.push_back(SVector<2>(std::pow(10, x), 1.0));
        }
        kcv.fit(model, lambdas, RMSE(model));

        std::cout << kcv.optimum() << std::endl;
        // std::cout << kcv.avg_scores() << std::endl;
        file << kcv.optimum()[0] << '\n';

        // define MinRMSE
        // MinRMSE min_rmse;
        // std::vector<DVector<double>> lambdas;
        // for (double x = -2.0; x >= -4; x -= 0.25) {
        //     lambdas.push_back(SVector<2>(std::pow(10, x), 1.0));
        // }
        // std::cout << "debug" << std::endl;
        // min_rmse.fit(model, lambdas);
        // std::cout << "debug" << std::endl;

        // std::cout << min_rmse.optimum() << std::endl;
        // std::cout << min_rmse.scores() << std::endl;
        //file << min_rmse.optimum()[0] << '\n';
    }

    file.close();
}

TEST(kcv_strpde_test_error, NLstrpde_monolithic_5error) {
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
    DMatrix<double> locs  = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_5/space_locs.csv");
    DMatrix<double> IC    = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_5/IC.csv");
    // define regularizing PDE
    double alpha = 7;
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 15/(5 + ff[0]);};
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

    std::ofstream file("lambdas5_NLSTRPDE_wrongPDE_monolithic.txt");    //it will be exported in the current build directory

    int n_obs = 30;
    for (size_t i=1; i<=n_obs; ++i) {
        std::cout << "Iteration " << i << std::endl;
        // define model
        STRPDE_NonLinear<SpaceTimeParabolic, fdapde::iterative> model(pde, Sampling::pointwise);
        model.set_spatial_locations(locs);
        // set model's data
        std::string filename = "../data/models/strpde_nonlinear/test nuovo/test_nuovo_5/y" + std::to_string(i) + ".csv";
        DMatrix<double> y = read_csv<double>(filename);
        BlockFrame<double, int> df;
        df.stack(OBSERVATIONS_BLK, y);
        model.set_data(df);
        // set parameters for iterative method
        model.set_tolerance(1e-5, 1e-8);
        model.set_max_iter(50);
        // initialize smoothing problem
        model.init();

        // define KCV engine and search for best lambda which minimizes the model's RMSE
        std::size_t n_folds = 5;
        int seed = 435642;
        KCV kcv(n_folds, seed, time_mesh.size(), false);
        std::vector<DVector<double>> lambdas;
        for (double x = -5; x <= 0; x += 0.25) {
            lambdas.push_back(SVector<2>(std::pow(10, x), 1.0));
        }
        kcv.fit(model, lambdas, RMSE(model));

        std::cout << kcv.optimum() << std::endl;
        // std::cout << kcv.avg_scores() << std::endl;
        file << kcv.optimum()[0] << '\n';
    }

    file.close();
}



TEST(kcv_strpde_test_error, strpde_NLiterative_10error) {
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
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 15/(5 + ff[0]);};
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

    std::ofstream file("lambdas10_NLSTRPDE_wrongPDE_iterative.txt");    //it will be exported in the current build directory

    int n_obs = 30;
    for (size_t i=1; i<=n_obs; ++i) {
        std::cout << "Iteration " << i << std::endl;
        // define model
        STRPDE_NonLinear<SpaceTimeParabolic, fdapde::iterative> model(pde, Sampling::pointwise);
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

        // define KCV engine and search for best lambda which minimizes the model's RMSE
        std::size_t n_folds = 5;
        int seed = 435642;
        KCV kcv(n_folds, seed, time_mesh.size(), false);
        std::vector<DVector<double>> lambdas;
        for (double x = -5; x <= 0; x += 0.25) {
            lambdas.push_back(SVector<2>(std::pow(10, x), 1.0));
        }
        kcv.fit(model, lambdas, RMSE(model));

        std::cout << kcv.optimum() << std::endl;
        // std::cout << kcv.avg_scores() << std::endl;
        file << kcv.optimum()[0] << '\n';

        // define MinRMSE
        // MinRMSE min_rmse;
        // std::vector<DVector<double>> lambdas;
        // for (double x = -2.0; x >= -4; x -= 0.25) {
        //     lambdas.push_back(SVector<2>(std::pow(10, x), 1.0));
        // }
        // std::cout << "debug" << std::endl;
        // min_rmse.fit(model, lambdas);
        // std::cout << "debug" << std::endl;

        // std::cout << min_rmse.optimum() << std::endl;
        // std::cout << min_rmse.scores() << std::endl;
        //file << min_rmse.optimum()[0] << '\n';
    }

    file.close();
}

TEST(kcv_strpde_test_error, NLstrpde_iterativeEI_10error) {
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
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 15/(5 + ff[0]);};
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

    std::ofstream file("lambdas10_NLSTRPDE_wrongPDE_iterative_EI.txt");    //it will be exported in the current build directory

    int n_obs = 30;
    for (size_t i=1; i<=n_obs; ++i) {
        std::cout << "Iteration " << i << std::endl;
        // define model
        STRPDE_NonLinear<SpaceTimeParabolic, fdapde::iterative_EI> model(pde, Sampling::pointwise);
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

        // define KCV engine and search for best lambda which minimizes the model's RMSE
        std::size_t n_folds = 5;
        int seed = 435642;
        KCV kcv(n_folds, seed, time_mesh.size(), false);
        std::vector<DVector<double>> lambdas;
        for (double x = -5; x <= 0; x += 0.25) {
            lambdas.push_back(SVector<2>(std::pow(10, x), 1.0));
        }
        kcv.fit(model, lambdas, RMSE(model));

        std::cout << kcv.optimum() << std::endl;
        // std::cout << kcv.avg_scores() << std::endl;
        file << kcv.optimum()[0] << '\n';

        // define MinRMSE
        // MinRMSE min_rmse;
        // std::vector<DVector<double>> lambdas;
        // for (double x = -2.0; x >= -4; x -= 0.25) {
        //     lambdas.push_back(SVector<2>(std::pow(10, x), 1.0));
        // }
        // std::cout << "debug" << std::endl;
        // min_rmse.fit(model, lambdas);
        // std::cout << "debug" << std::endl;

        // std::cout << min_rmse.optimum() << std::endl;
        // std::cout << min_rmse.scores() << std::endl;
        //file << min_rmse.optimum()[0] << '\n';
    }

    file.close();
}

TEST(kcv_strpde_test_error, NLstrpde_monolithic_10error) {
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
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 15/(5 + ff[0]);};
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

    std::ofstream file("lambdas10_NLSTRPDE_wrongPDE_monolithic.txt");    //it will be exported in the current build directory

    int n_obs = 30;
    for (size_t i=1; i<=n_obs; ++i) {
        std::cout << "Iteration " << i << std::endl;
        // define model
        STRPDE_NonLinear<SpaceTimeParabolic, fdapde::iterative> model(pde, Sampling::pointwise);
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

        // define KCV engine and search for best lambda which minimizes the model's RMSE
        std::size_t n_folds = 5;
        int seed = 435642;
        KCV kcv(n_folds, seed, time_mesh.size(), false);
        std::vector<DVector<double>> lambdas;
        for (double x = -5; x <= 0; x += 0.25) {
            lambdas.push_back(SVector<2>(std::pow(10, x), 1.0));
        }
        kcv.fit(model, lambdas, RMSE(model));

        std::cout << kcv.optimum() << std::endl;
        // std::cout << kcv.avg_scores() << std::endl;
        file << kcv.optimum()[0] << '\n';
    }

    file.close();
}



TEST(kcv_strpde_test_error, strpde_NLiterative_20error) {
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
    DMatrix<double> locs  = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_20/space_locs.csv");
    DMatrix<double> IC    = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_20/IC.csv");
    // define regularizing PDE
    double alpha = 7;
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 15/(5 + ff[0]);};
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

    std::ofstream file("lambdas20_NLSTRPDE_wrongPDE_iterative.txt");    //it will be exported in the current build directory

    int n_obs = 30;
    for (size_t i=1; i<=n_obs; ++i) {
        std::cout << "Iteration " << i << std::endl;
        // define model
        STRPDE_NonLinear<SpaceTimeParabolic, fdapde::iterative> model(pde, Sampling::pointwise);
        model.set_spatial_locations(locs);
        // set model's data
        std::string filename = "../data/models/strpde_nonlinear/test nuovo/test_nuovo_20/y" + std::to_string(i) + ".csv";
        DMatrix<double> y = read_csv<double>(filename);
        BlockFrame<double, int> df;
        df.stack(OBSERVATIONS_BLK, y);
        model.set_data(df);
        // set parameters for iterative method
        model.set_tolerance(1e-5, 1e-8);
        model.set_max_iter(50);
        // initialize smoothing problem
        model.init();

        // define KCV engine and search for best lambda which minimizes the model's RMSE
        std::size_t n_folds = 5;
        int seed = 435642;
        KCV kcv(n_folds, seed, time_mesh.size(), false);
        std::vector<DVector<double>> lambdas;
        for (double x = -5; x <= 0; x += 0.25) {
            lambdas.push_back(SVector<2>(std::pow(10, x), 1.0));
        }
        kcv.fit(model, lambdas, RMSE(model));

        std::cout << kcv.optimum() << std::endl;
        // std::cout << kcv.avg_scores() << std::endl;
        file << kcv.optimum()[0] << '\n';

        // define MinRMSE
        // MinRMSE min_rmse;
        // std::vector<DVector<double>> lambdas;
        // for (double x = -2.0; x >= -4; x -= 0.25) {
        //     lambdas.push_back(SVector<2>(std::pow(10, x), 1.0));
        // }
        // std::cout << "debug" << std::endl;
        // min_rmse.fit(model, lambdas);
        // std::cout << "debug" << std::endl;

        // std::cout << min_rmse.optimum() << std::endl;
        // std::cout << min_rmse.scores() << std::endl;
        //file << min_rmse.optimum()[0] << '\n';
    }

    file.close();
}

TEST(kcv_strpde_test_error, NLstrpde_iterativeEI_20error) {
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
    DMatrix<double> locs  = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_20/space_locs.csv");
    DMatrix<double> IC    = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_20/IC.csv");
    // define regularizing PDE
    double alpha = 7;
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 15/(5 + ff[0]);};
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

    std::ofstream file("lambdas20_NLSTRPDE_wrongPDE_iterative_EI.txt");    //it will be exported in the current build directory

    int n_obs = 30;
    for (size_t i=1; i<=n_obs; ++i) {
        std::cout << "Iteration " << i << std::endl;
        // define model
        STRPDE_NonLinear<SpaceTimeParabolic, fdapde::iterative_EI> model(pde, Sampling::pointwise);
        model.set_spatial_locations(locs);
        // set model's data
        std::string filename = "../data/models/strpde_nonlinear/test nuovo/test_nuovo_20/y" + std::to_string(i) + ".csv";
        DMatrix<double> y = read_csv<double>(filename);
        BlockFrame<double, int> df;
        df.stack(OBSERVATIONS_BLK, y);
        model.set_data(df);
        // set parameters for iterative method
        model.set_tolerance(1e-5, 1e-8);
        model.set_max_iter(50);
        // initialize smoothing problem
        model.init();

        // define KCV engine and search for best lambda which minimizes the model's RMSE
        std::size_t n_folds = 5;
        int seed = 435642;
        KCV kcv(n_folds, seed, time_mesh.size(), false);
        std::vector<DVector<double>> lambdas;
        for (double x = -5; x <= 0; x += 0.25) {
            lambdas.push_back(SVector<2>(std::pow(10, x), 1.0));
        }
        kcv.fit(model, lambdas, RMSE(model));

        std::cout << kcv.optimum() << std::endl;
        // std::cout << kcv.avg_scores() << std::endl;
        file << kcv.optimum()[0] << '\n';

        // define MinRMSE
        // MinRMSE min_rmse;
        // std::vector<DVector<double>> lambdas;
        // for (double x = -2.0; x >= -4; x -= 0.25) {
        //     lambdas.push_back(SVector<2>(std::pow(10, x), 1.0));
        // }
        // std::cout << "debug" << std::endl;
        // min_rmse.fit(model, lambdas);
        // std::cout << "debug" << std::endl;

        // std::cout << min_rmse.optimum() << std::endl;
        // std::cout << min_rmse.scores() << std::endl;
        //file << min_rmse.optimum()[0] << '\n';
    }

    file.close();
}

TEST(kcv_strpde_test_error, NLstrpde_monolithic_20error) {
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
    DMatrix<double> locs  = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_20/space_locs.csv");
    DMatrix<double> IC    = read_csv<double>("../data/models/strpde_nonlinear/test nuovo/test_nuovo_20/IC.csv");
    // define regularizing PDE
    double alpha = 7;
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 15/(5 + ff[0]);};
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

    std::ofstream file("lambdas20_NLSTRPDE_wrongPDE_monolithic.txt");    //it will be exported in the current build directory

    int n_obs = 30;
    for (size_t i=1; i<=n_obs; ++i) {
        std::cout << "Iteration " << i << std::endl;
        // define model
        STRPDE_NonLinear<SpaceTimeParabolic, fdapde::iterative> model(pde, Sampling::pointwise);
        model.set_spatial_locations(locs);
        // set model's data
        std::string filename = "../data/models/strpde_nonlinear/test nuovo/test_nuovo_20/y" + std::to_string(i) + ".csv";
        DMatrix<double> y = read_csv<double>(filename);
        BlockFrame<double, int> df;
        df.stack(OBSERVATIONS_BLK, y);
        model.set_data(df);
        // set parameters for iterative method
        model.set_tolerance(1e-5, 1e-8);
        model.set_max_iter(50);
        // initialize smoothing problem
        model.init();

        // define KCV engine and search for best lambda which minimizes the model's RMSE
        std::size_t n_folds = 5;
        int seed = 435642;
        KCV kcv(n_folds, seed, time_mesh.size(), false);
        std::vector<DVector<double>> lambdas;
        for (double x = -5; x <= 0; x += 0.25) {
            lambdas.push_back(SVector<2>(std::pow(10, x), 1.0));
        }
        kcv.fit(model, lambdas, RMSE(model));

        std::cout << kcv.optimum() << std::endl;
        // std::cout << kcv.avg_scores() << std::endl;
        file << kcv.optimum()[0] << '\n';
    }

    file.close();
}