// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Vincent Lejeune
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BLOCK_HOUSEHOLDER_H
#define EIGEN_BLOCK_HOUSEHOLDER_H

// This file contains some helper function to deal with block householder reflectors

namespace Eigen { 

namespace internal {
  
/** \internal */
// template<typename TriangularFactorType,typename VectorsType,typename CoeffsType>
// void make_block_householder_triangular_factor(TriangularFactorType& triFactor, const VectorsType& vectors, const CoeffsType& hCoeffs)
// {
//   typedef typename VectorsType::Scalar Scalar;
//   const Index nbVecs = vectors.cols();
//   eigen_assert(triFactor.rows() == nbVecs && triFactor.cols() == nbVecs && vectors.rows()>=nbVecs);
// 
//   for(Index i = 0; i < nbVecs; i++)
//   {
//     Index rs = vectors.rows() - i;
//     // Warning, note that hCoeffs may alias with vectors.
//     // It is then necessary to copy it before modifying vectors(i,i). 
//     typename CoeffsType::Scalar h = hCoeffs(i);
//     // This hack permits to pass trough nested Block<> and Transpose<> expressions.
//     Scalar *Vii_ptr = const_cast<Scalar*>(vectors.data() + vectors.outerStride()*i + vectors.innerStride()*i);
//     Scalar Vii = *Vii_ptr;
//     *Vii_ptr = Scalar(1);
//     triFactor.col(i).head(i).noalias() = -h * vectors.block(i, 0, rs, i).adjoint()
//                                        * vectors.col(i).tail(rs);
//     *Vii_ptr = Vii;
//     // FIXME add .noalias() once the triangular product can work inplace
//     triFactor.col(i).head(i) = triFactor.block(0,0,i,i).template triangularView<Upper>()
//                              * triFactor.col(i).head(i);
//     triFactor(i,i) = hCoeffs(i);
//   }
// }

/** \internal */
// This variant avoid modifications in vectors
template<typename TriangularFactorType,typename VectorsType,typename CoeffsType>
void make_block_householder_triangular_factor(TriangularFactorType& triFactor, const VectorsType& vectors, const CoeffsType& hCoeffs)
{
  const Index nbVecs = vectors.cols();
  eigen_assert(triFactor.rows() == nbVecs && triFactor.cols() == nbVecs && vectors.rows()>=nbVecs);

  for(Index i = nbVecs-1; i >=0 ; --i)
  {
    Index rs = vectors.rows() - i - 1;
    Index rt = nbVecs-i-1;

    if(rt>0)
    {
      triFactor.row(i).tail(rt).noalias() = -hCoeffs(i) * vectors.col(i).tail(rs).