// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<typename MatrixType> void product_selfadjoint(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  typedef Matrix<Scalar, 1, MatrixType::RowsAtCompileTime> RowVectorType;

  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, Dynamic, RowMajor> RhsMatrixType;

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3;
  VectorType v1 = VectorType::Random(rows),
             v2 = VectorType::Random(rows),
             v3(rows);
  RowVectorType r1 = RowVectorType::Random(rows),
                r2 = RowVectorType::Random(rows);
  RhsMatrixType m4 = RhsMatrixType::Random(rows,10);

  Scalar s1 = internal::random<Scalar>(),
         s2 = internal::random<Scalar>(),
         s3 = internal::random<Scalar>();

  m1 = (m1.adjoin