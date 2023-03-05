// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <sstream>
#include <string>
#include <Eigen/CXX11/Tensor>


template<int DataLayout>
static void test_output_0d()
{
  Tensor<int, 0, DataLayout> tensor;
  tensor() = 123;

  std::stringstream os;
  os << tensor;

  std::string expected("123");
  VERIFY_IS_EQUAL(std::string(os.str()), expected);
}


template<int DataLayout>
static void test_output_1d()
{
  Tensor<int, 1, DataLayout> tensor(5);
  for (int i = 0; i < 5; ++i) {
    tensor(i) = i;
  }

  std::stringstream os;
  os << tensor;

  std::string expected("0\n1\n2\n3\n4");
  VERIFY_IS_EQUAL(std::string(os.str()), expected);

  Eigen::Tensor<double,1,DataLayout> empty_tensor(0);
  std::stringstream empty_os;
  empty_os << empty_tensor;
  std::string empty_string;
  VERIFY_IS_EQUAL(std::string(empty_os.str()), empty_string);
}


template<int DataLayout>
static void test_output_2d()
{
  Tensor<int, 2, DataLayout> tensor(5, 3);
  for (int i = 0; i 