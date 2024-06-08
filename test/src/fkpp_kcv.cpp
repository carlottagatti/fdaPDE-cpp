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
using fdapde::core::Mesh1D;
using fdapde::core::reaction;
using fdapde::core::NonLinearReaction;
using fdapde::core::LagrangianBasis;
using fdapde::core::non_linear_op;

#include "../../fdaPDE/models/regression/srpde.h"
#include "../../fdaPDE/models/regression/strpde_nonlinear.h"
#include "../../fdaPDE/models/regression/qsrpde.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::SRPDE;
using fdapde::models::STRPDE_NonLinear;
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

// test 1 - method strpde monolithic
//    domain:       interval [0, 1900]
//    sampling:     locations = nodes
//    penalization: fkpp equation
//    covariates:   no
//    BC:           homogeneous Neumann
//    order FE:     1
//    time penalization: parabolic (monolithic solver)
TEST(fkpp_kcv_test, strpde_monolithic) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(4);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 12*(i+1);
    // define spatial domain
    Mesh1D domain(0.0, 1900, 37);

    // import data from files
    Mesh1D mesh_locations(25, 1875, 37);
    DMatrix<double> locs = mesh_locations.nodes();
    DMatrix<double> y    = read_csv<double>("../data/models/strpde_nonlinear/fkpp data/y2.csv");
    DMatrix<double> IC   = read_csv<double>("../data/models/strpde_nonlinear/fkpp data/IC2.csv");
    // define regularizing PDE
    double lambda = 0.044;  // cell proliferation rate [1/h]
    double D = 250;         // cell diffusivity [(\mu m)^2 / h]
    double K = 1.7 * 1e-3;  // carrying capacity density [cell/(\mu m)^2]
    // define the regolarizing PDE
    auto L = dt<FEM>() - D*laplacian<FEM>() + reaction<FEM>(lambda);
    PDE<decltype(domain), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain, time_mesh, L);
    pde.set_initial_condition(IC);
    //set forcing
    auto forcing_expr = [&](SVector<1> x, double t) -> double {
        return 0;
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);

    // save the results in a file
    std::ofstream file("fkpp2_lambdas_STRPDE_monolithic.txt");    //it will be exported in the current build directory

    int n_obs = 1;
    for (size_t i=1; i<=n_obs; ++i) {
        // define model
        STRPDE<SpaceTimeParabolic, fdapde::monolithic> model(pde, Sampling::pointwise);
        model.set_spatial_locations(locs);
        // set model's data
        BlockFrame<double, int> df;
        df.stack(OBSERVATIONS_BLK, y);
        model.set_data(df);
        // initialize smoothing problem
        model.init();

        // define KCV engine and search for best lambda which minimizes the model's RMSE
        std::size_t n_folds = 5;
        int seed = 435642;
        KCV kcv(n_folds, seed, time_mesh.size(), false);
        std::vector<DVector<double>> lambdas;
        for (double x = -5.0; x <= 0; x += 0.25) {
            lambdas.push_back(SVector<2>(std::pow(10, x), 1.0));
        }
        kcv.fit(model, lambdas, RMSE(model));

        std::cout << kcv.optimum() << std::endl;
        std::cout << kcv.avg_scores() << std::endl;
        file << kcv.optimum()[0] << '\n';
    }

    file.close();
}


// test 1 - method strpde iterative
//    domain:       interval [0, 1900]
//    sampling:     locations = nodes
//    penalization: fkpp equation
//    covariates:   no
//    BC:           homogeneous Neumann
//    order FE:     1
//    time penalization: parabolic (itervative solver)
TEST(fkpp_kcv_test, strpde_iterative) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(4);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 12*(i+1);
    // define spatial domain
    Mesh1D domain(0.0, 1900, 37);

    // import data from files
    Mesh1D mesh_locations(25, 1875, 37);
    DMatrix<double> locs = mesh_locations.nodes();
    DMatrix<double> y    = read_csv<double>("../data/models/strpde_nonlinear/fkpp data/y2.csv");
    DMatrix<double> IC   = read_csv<double>("../data/models/strpde_nonlinear/fkpp data/IC2.csv");
    // define regularizing PDE
    double lambda = 0.044;  // cell proliferation rate [1/h]
    double D = 250;         // cell diffusivity [(\mu m)^2 / h]
    double K = 1.7 * 1e-3;  // carrying capacity density [cell/(\mu m)^2]
    // define the regolarizing PDE
    auto L = dt<FEM>() - D*laplacian<FEM>() + reaction<FEM>(lambda);
    PDE<decltype(domain), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain, time_mesh, L);
    pde.set_initial_condition(IC);
    //set forcing
    auto forcing_expr = [&](SVector<1> x, double t) -> double {
        return 0;
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);

    // save the results in a file
    std::ofstream file("fkpp2_lambdas_STRPDE_iterative.txt");    //it will be exported in the current build directory

    int n_obs = 1;
    for (size_t i=1; i<=n_obs; ++i) {
        // define model
        STRPDE<SpaceTimeParabolic, fdapde::iterative> model(pde, Sampling::pointwise);
        model.set_spatial_locations(locs);
        // set model's data
        BlockFrame<double, int> df;
        df.stack(OBSERVATIONS_BLK, y);
        model.set_data(df);
        // initialize smoothing problem
        model.init();

        // define KCV engine and search for best lambda which minimizes the model's RMSE
        std::size_t n_folds = 5;
        int seed = 435642;
        KCV kcv(n_folds, seed, time_mesh.size(), false);
        std::vector<DVector<double>> lambdas;
        for (double x = -5.0; x <= 0; x += 0.25) {
            lambdas.push_back(SVector<2>(std::pow(10, x), 1.0));
        }
        kcv.fit(model, lambdas, RMSE(model));

        std::cout << kcv.optimum() << std::endl;
        std::cout << kcv.avg_scores() << std::endl;
        file << kcv.optimum()[0] << '\n';
    }

    file.close();
}

// test 1 - method iterative_EI
//    domain:       interval [0, 1900]
//    sampling:     locations = nodes
//    penalization: fkpp equation
//    covariates:   no
//    BC:           homogeneous Neumann
//    order FE:     1
//    time penalization: parabolic (iterative_EI solver)
TEST(fkpp_kcv_test, nlstrpde_iterative_EI) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(4);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 12*(i+1);
    // define spatial domain
    Mesh1D domain(0.0, 1900, 37);
    // import data from files
    Mesh1D mesh_locations(25, 1875, 37);
    DMatrix<double> locs = mesh_locations.nodes();
    DMatrix<double> y    = read_csv<double>("../data/models/strpde_nonlinear/fkpp data/y2.csv");
    DMatrix<double> IC   = read_csv<double>("../data/models/strpde_nonlinear/fkpp data/IC2.csv");
    // define regularizing PDE
    double lambda = 0.044;  // cell proliferation rate [1/h]
    double D = 250;         // cell diffusivity [(\mu m)^2 / h]
    double K = 1.7 * 1e-3;  // carrying capacity density [cell/(\mu m)^2]
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) ->double {return lambda*(1 - ff[0]/K);};
    NonLinearReaction<1, LagrangianBasis<decltype(domain),1>::ReferenceBasis> non_linear_reaction(h_);
    auto L = dt<FEM>() - D*laplacian<FEM>() - non_linear_op<FEM>(non_linear_reaction);
    PDE<decltype(domain), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain, time_mesh, L, h_);
    pde.set_initial_condition(IC);
    //set forcing
    auto forcing_expr = [&](SVector<1> x, double t) -> double {
        return 0;
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);

    std::ofstream file("fkpp2_lambdas_NLSTRPDE_iterativeEI.txt");    //it will be exported in the current build directory

    int n_obs = 1;
    for (size_t i=1; i<=n_obs; ++i) {
        // define model
        STRPDE_NonLinear<SpaceTimeParabolic, fdapde::iterative_EI> model(pde, Sampling::pointwise);
        model.set_spatial_locations(locs);
        // set model's data
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
        std::cout << kcv.avg_scores() << std::endl;
        file << kcv.optimum()[0] << '\n';
    }

    file.close();
}

// test 1 - method iterative
//    domain:       interval [0, 1900]
//    sampling:     locations = nodes
//    penalization: fkpp equation
//    covariates:   no
//    BC:           homogeneous Neumann
//    order FE:     1
//    time penalization: parabolic (iterative solver)
TEST(fkpp_kcv_test, nlstrpde_iterative) {
    // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(4);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 12*(i+1);
    // define spatial domain
    Mesh1D domain(0.0, 1900, 37);
    // import data from files
    Mesh1D mesh_locations(25, 1875, 37);
    DMatrix<double> locs = mesh_locations.nodes();
    DMatrix<double> y    = read_csv<double>("../data/models/strpde_nonlinear/fkpp data/y2.csv");
    DMatrix<double> IC   = read_csv<double>("../data/models/strpde_nonlinear/fkpp data/IC2.csv");
    // define regularizing PDE
    double lambda = 0.044;  // cell proliferation rate [1/h]
    double D = 250;         // cell diffusivity [(\mu m)^2 / h]
    double K = 1.7 * 1e-3;  // carrying capacity density [cell/(\mu m)^2]
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) ->double {return lambda*(1 - ff[0]/K);};
    NonLinearReaction<1, LagrangianBasis<decltype(domain),1>::ReferenceBasis> non_linear_reaction(h_);
    auto L = dt<FEM>() - D*laplacian<FEM>() - non_linear_op<FEM>(non_linear_reaction);
    PDE<decltype(domain), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain, time_mesh, L, h_);
    pde.set_initial_condition(IC);
    //set forcing
    auto forcing_expr = [&](SVector<1> x, double t) -> double {
        return 0;
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);

    std::ofstream file("fkpp2_lambdas_NLSTRPDE_iterative.txt");    //it will be exported in the current build directory

    int n_obs = 1;
    for (size_t i=1; i<=n_obs; ++i) {
        // define model
        STRPDE_NonLinear<SpaceTimeParabolic, fdapde::iterative> model(pde, Sampling::pointwise);
        model.set_spatial_locations(locs);
        // set model's data
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
        std::cout << kcv.avg_scores() << std::endl;
        file << kcv.optimum()[0] << '\n';
    }

    file.close();
}


// test 1 - method monolithic
//    domain:       interval [0, 1900]
//    sampling:     locations = nodes
//    penalization: fkpp euqation
//    covariates:   no
//    BC:           homogeneous Neumann
//    order FE:     1
//    time penalization: parabolic (monolithic solver)
TEST(fkpp_kcv_test, nlstrpde_monolithic) {
   // define temporal domain
    DVector<double> time_mesh;
    time_mesh.resize(4);
    for (std::size_t i = 0; i < time_mesh.size(); ++i) time_mesh[i] = 12*(i+1);
    // define spatial domain
    Mesh1D domain(0.0, 1900, 37);
    // import data from files
    Mesh1D mesh_locations(25, 1875, 37);
    DMatrix<double> locs = mesh_locations.nodes();
    DMatrix<double> y    = read_csv<double>("../data/models/strpde_nonlinear/fkpp data/y2.csv");
    DMatrix<double> IC   = read_csv<double>("../data/models/strpde_nonlinear/fkpp data/IC2.csv");
    // define regularizing PDE
    double lambda = 0.044;  // cell proliferation rate [1/h]
    double D = 250;         // cell diffusivity [(\mu m)^2 / h]
    double K = 1.7 * 1e-3;  // carrying capacity density [cell/(\mu m)^2]
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) ->double {return lambda*(1 - ff[0]/K);};
    NonLinearReaction<1, LagrangianBasis<decltype(domain),1>::ReferenceBasis> non_linear_reaction(h_);
    auto L = dt<FEM>() - D*laplacian<FEM>() - non_linear_op<FEM>(non_linear_reaction);
    PDE<decltype(domain), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain, time_mesh, L, h_);
    pde.set_initial_condition(IC);
    //set forcing
    auto forcing_expr = [&](SVector<1> x, double t) -> double {
        return 0;
    };
    DMatrix<double> quadrature_nodes = pde.force_quadrature_nodes();
    DMatrix<double> u(quadrature_nodes.rows(), time_mesh.rows());
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < time_mesh.rows(); ++j) { u(i, j) = forcing_expr(quadrature_nodes.row(i), time_mesh(j)); }
    }
    pde.set_forcing(u);

    std::ofstream file("fkpp2_lambdas_NLSTRPDE_monolithic.txt");    //it will be exported in the current build directory

    int n_obs = 1;
    for (size_t i=1; i<=n_obs; ++i) {
        // define model
        STRPDE_NonLinear<SpaceTimeParabolic, fdapde::monolithic> model(pde, Sampling::pointwise);
        model.set_spatial_locations(locs);
        // set model's data
        BlockFrame<double, int> df;
        df.stack(OBSERVATIONS_BLK, y);
        model.set_data(df);
        // set parameters for iterative method
        model.set_tolerance(1e-5);
        model.set_max_iter(15);
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
        std::cout << kcv.avg_scores() << std::endl;
        file << kcv.optimum()[0] << '\n';
    }

    file.close();
}