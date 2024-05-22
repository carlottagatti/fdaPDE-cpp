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
using fdapde::core::FEM;
using fdapde::core::fem_order;
using fdapde::core::laplacian;
using fdapde::core::DiscretizedMatrixField;
using fdapde::core::PDE;
using fdapde::core::DiscretizedVectorField;

#include "../../fdaPDE/models/regression/srpde.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::SRPDE;
using fdapde::models::Sampling;

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;

// test 1
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   no
//    BC:           no
//    order FE:     1
TEST(srpde_test, laplacian_nonparametric_samplingatnodes) {
    // define domain 
    MeshLoader<Mesh2D> domain("unit_square");
    // define boundary conditions matrix: 0=Dirichlet, 1=Neumann, 2=Robin
    /* DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1) ; // has all zeros
    for (size_t j=1; j<59; ++j) boundary_matrix(j, 0) = 2;
    for (size_t j=3541; j<3599; ++j) boundary_matrix(j, 0) = 2; */
    // import data from files
    DMatrix<double> y = read_csv<double>("../data/models/srpde/2D_test1/y.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 6, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L);
    problem.set_forcing(u);

    // define Dirichlet boundary conditions
    /* DMatrix<double> nodes_ = problem.dof_coords();
    DMatrix<double> dirichlet_bc(nodes_.rows(), 1);
    for (int i = 0; i < nodes_.rows(); ++i) {
        dirichlet_bc(i) =  25.0;
    }
    // set dirichlet boundary data as data of the problem
    problem.set_dirichlet_bc(dirichlet_bc); */

    // define Neumann boundary conditions
    /* DMatrix<double> boundary_quadrature_nodes = problem.boundary_quadrature_nodes();
    DMatrix<double> f_neumann(boundary_quadrature_nodes.rows(), 1);
    for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
        f_neumann(i) = 1.0;
    }
    problem.set_neumann_bc(f_neumann); */

    // define Robin boundary conditions
    /* DMatrix<double> boundary_quadrature_nodes = problem.boundary_quadrature_nodes();
    DMatrix<double> f_robin(boundary_quadrature_nodes.rows(), 1);
    for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
        f_robin(i) = 2.0;
    }
    DVector<double> robin_constants(2,1);
    robin_constants << 2, 5;
    problem.set_robin_bc(f_robin, robin_constants); */

    // define model
    double lambda = 5.623413 * std::pow(0.1, 5);
    SRPDE model(problem, Sampling::mesh_nodes);
    model.set_lambda_D(lambda);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();
    // test correctness
    EXPECT_TRUE(almost_equal(model.f()  , "../data/models/srpde/2D_test1/sol.mtx"));
}

// test 2
//    domain:       c-shaped
//    sampling:     locations != nodes
//    penalization: simple laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
TEST(srpde_test, laplacian_semiparametric_samplingatlocations) {
    // define domain
    MeshLoader<Mesh2D> domain("c_shaped");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/srpde/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/srpde/2D_test2/y.csv");
    DMatrix<double> X    = read_csv<double>("../data/models/srpde/2D_test2/X.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 6, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L); problem.set_forcing(u);
    // define statistical model
    double lambda = 0.2201047;
    SRPDE model(problem, Sampling::pointwise);
    model.set_lambda_D(lambda);
    model.set_spatial_locations(locs);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    df.insert(DESIGN_MATRIX_BLK, X);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();
    // test correctness
    EXPECT_TRUE(almost_equal(model.f()   , "../data/models/srpde/2D_test2/sol.mtx" ));
    EXPECT_TRUE(almost_equal(model.beta(), "../data/models/srpde/2D_test2/beta.mtx"));
}

// test 3
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: costant coefficients PDE
//    covariates:   no
//    BC:           no
//    order FE:     1
TEST(srpde_test, costantcoefficientspde_nonparametric_samplingatnodes) {
    // define domain
    MeshLoader<Mesh2D> domain("unit_square");
    // import data from files
    DMatrix<double> y = read_csv<double>("../data/models/srpde/2D_test3/y.csv");
    // define regularizing PDE
    SMatrix<2> K;
    K << 1, 0, 0, 4;
    auto L = -diffusion<FEM>(K);   // anisotropic diffusion
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 6, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L); problem.set_forcing(u);
    // define  model
    double lambda = 10;
    SRPDE model(problem, Sampling::mesh_nodes);
    model.set_lambda_D(lambda);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();
    // test correctness
    EXPECT_TRUE(almost_equal(model.f() , "../data/models/srpde/2D_test3/sol.mtx"));
}

// test 4
//    domain:       quasicircular domain
//    sampling:     areal
//    penalization: non-costant coefficients PDE
//    covariates:   no
//    BC:           yes
//    order FE:     1
/*
TEST(srpde_test, noncostantcoefficientspde_nonparametric_samplingareal) {
    // define domain
    MeshLoader<Mesh2D> domain("quasi_circle");
    // import data from files
    DMatrix<double, Eigen::RowMajor> K_data  = read_csv<double>("../data/models/srpde/2D_test4/K.csv");
    DMatrix<double, Eigen::RowMajor> b_data  = read_csv<double>("../data/models/srpde/2D_test4/b.csv");
    DMatrix<double> subdomains = read_csv<double>("../data/models/srpde/2D_test4/incidence_matrix.csv");
    DMatrix<double> u = read_csv<double>("../data/models/srpde/2D_test4/force.csv");
    DMatrix<double> y = read_csv<double>("../data/models/srpde/2D_test4/y.csv"    );
    // define regularizing PDE
    DiscretizedMatrixField<2, 2, 2> K(K_data);
    DiscretizedVectorField<2, 2> b(b_data);
    auto L = -diffusion<FEM>(K) + advection<FEM>(b);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L); problem.set_forcing(u);
    // define model
    double lambda = std::pow(0.1, 3);
    SRPDE model(problem, Sampling::areal);
    model.set_lambda_D(lambda);
    model.set_spatial_locations(subdomains);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();
    // test correctness
    EXPECT_TRUE(almost_equal(model.f() , "../data/models/srpde/2D_test4/sol.mtx"));
}
*/
// test 5
//    domain:       c-shaped surface
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   no
//    BC:           no
//    order FE:     1
TEST(srpde_test, laplacian_nonparametric_samplingatnodes_surface) {
    // define domain 
    MeshLoader<SurfaceMesh> domain("c_shaped_surface");
    // import data from files
    DMatrix<double> y = read_csv<double>("../data/models/srpde/2D_test5/y.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 6, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L); problem.set_forcing(u);
    // define model
    double lambda = 1e-2;
    SRPDE model(problem, Sampling::mesh_nodes);
    model.set_lambda_D(lambda);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve smoothing problem
    model.init();
    model.solve();
    // test correctness
    EXPECT_TRUE(almost_equal(model.f()  , "../data/models/srpde/2D_test5/sol.mtx"));
}
