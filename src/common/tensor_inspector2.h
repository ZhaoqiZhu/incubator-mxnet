/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2019 by Contributors
 * \file tensor_inspector.h
 * \brief utility to inspect tensor objects
 * \author Zhaoqi Zhu
 */

#ifndef MXNET_COMMON_TENSOR_INSPECTOR_H_
#define MXNET_COMMON_TENSOR_INSPECTOR_H_

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include "../../3rdparty/mshadow/mshadow/base.h"
#include "../../tests/cpp/include/test_util.h"

namespace mxnet {

namespace static_if_detail {

struct identity {
    template<typename T>
    T operator()(T&& x) const {
        return std::forward<T>(x);
    }
};

template<bool Cond>
struct statement {
    template<typename F>
    void then(const F& f){
        f();
    }

    template<typename F>
    void else_(const F&){}
};

template<>
struct statement<false> {
    template<typename F>
    void then(const F&){}

    template<typename F>
    void else_(const F& f){
        f();
    }
};

} //end of namespace static_if_detail

template<bool Cond, typename F>
static_if_detail::statement<Cond> static_if(F const& f){
    static_if_detail::statement<Cond> if_;
    if_.then(f);
    return if_;
}

#define IF_FLOAT_TYPE(DType, IF, ELSE) \
  static_if<std::is_same<DType, float>::value>([]{ \
    {__VA_ARGS__} \
  }).else_([] { \
    LOG(WARNING) << "FiniteChecker only applies to float types. " << \
        "Lambda will always return false."; \
  }); \

/*!
 * \brief This singleton struct mediates individual TensorInspector objects
 * so that we can control the global behavior from each of them
 */
struct InspectorManager {
  static InspectorManager* get() {
    static std::mutex mtx;
    static std::unique_ptr<InspectorManager> im = nullptr;
    if (!im) {
      std::unique_lock<std::mutex> lk(mtx);
      if (!im)
        im = std::make_unique<InspectorManager>();
    }
    return im.get();
  }
  /* !\brief mutex used to lock interactive_print() and check_value() */
  std::mutex mutex_;
  /* !\brief skip all interactive prints */
  bool interactive_print_skip_all_ = false;
  /* !\brief skip all value checks */
  bool check_value_skip_all_ = false;
  /* !\brief visit count for interactive print tags */
  std::unordered_map<std::string, int> interactive_print_tag_counter_;
  /* !\brief visit count for check value tags */
  std::unordered_map<std::string, int> check_value_tag_counter_;
  /* !\brief visit count for dump value tags */
  std::unordered_map<std::string, int> dump_value_tag_counter_;
};

/*!
 * \brief Enum for building value checkers for TensorInspector::check_value()
 */
enum CheckerType {
  NegativeChecker,  // check if is negative
  PositiveChecker,  // check if is positive
  ZeroChecker,  // check if is zero
  NaNChecker,  // check if is NaN, will always return false if DType is not a float type
  InfChecker,  // check if is infinity, will always return false if DType is not a float type
  PositiveInfChecker,  // check if is positive infinity,
                       // will always return false if DType is not a float type
  NegativeInfChecker,  // check if is nagative infinity,
                       // will always return false if DType is not a float type
  FiniteChecker,  // check if is finite, will always return false if DType is not a float type
  NormalChecker,  // check if is neither infinity nor NaN
};

template<typename DType MSHADOW_DEFAULT_DTYPE>
DType get_inf() {
  return (DType)1.0 / (DType)0.0;
}


/**
 *  _______                      _____                           _             
 * |__   __|                    |_   _|                         | |            
 *    | | ___ _ __  ___  ___  _ __| |  _ __  ___ _ __   ___  ___| |_ ___  _ __ 
 *    | |/ _ \ '_ \/ __|/ _ \| '__| | | '_ \/ __| '_ \ / _ \/ __| __/ _ \| '__|
 *    | |  __/ | | \__ \ (_) | | _| |_| | | \__ \ |_) |  __/ (__| || (_) | |   
 *    |_|\___|_| |_|___/\___/|_||_____|_| |_|___/ .__/ \___|\___|\__\___/|_|   
 *                                              | |                            
 *                                              |_|   
 */

/*!
 * \brief This class provides a unified interface to inspect the value of all data types
 * including Tensor, TBlob, and NDArray. If the tensor resides on GPU, then it will be
 * copied from GPU memory back to CPU memory to be operated on. Internally, all data types
 * are stored as a TBlob object tb_.
 */
class TensorInspector {
 private:
  /*!
   * \brief generate the tensor info, including data type and shape 
   * \tparam DType the data type
   * \tparam StreamType the type of the stream object
   * \param os stream object to output to
   */
  template<typename DType MSHADOW_DEFAULT_DTYPE, typename StreamType>
  inline void tensor_info_to_string(StreamType* os) {
    int dimension = tb_.ndim();
    *os << "<" << typeid(tb_.dptr<DType>()[0]).name() << " Tensor ";
    *os << tb_.shape_[0];
    for (int i = 1; i < dimension; i++) {
      *os << 'x' << tb_.shape_[i];
    }
    *os << ">" << std::endl;
  }

  /*!
   * \brief output the tensor info, including data type and shape 
   * \tparam DType the data type
   * \tparam StreamType the type of the stream object
   * \param os stream object to output to
   * \param shape the shape of the tensor
   */
  template<typename DType MSHADOW_DEFAULT_DTYPE, typename StreamType>
  inline void tensor_info_to_string(StreamType* os, const std::vector<int>& shape) {
    int dimension = shape.size();
    *os << "<" << typeid(tb_.dptr<DType>()[0]).name() << " Tensor ";
    *os << shape[0];
    for (int i = 1; i < dimension; i++) {
      *os << 'x' << shape[i];
    }
    *os << ">" << std::endl;
  }

  /*!
   * \brief output the tensor in a structed format 
   * \tparam DType the data type
   * \tparam StreamType the type of the stream object
   * \param os stream object to output to
   */
  template<typename DType MSHADOW_DEFAULT_DTYPE, typename StreamType>
  inline void to_string_helper(StreamType* os) {
#if MXNET_USE_CUDA
    if (tb_.dev_mask() == gpu::kDevMask) {
      TensorInspector(test::CAccessAsCPU(ctx_, tb_, false)(), ctx_)
          .to_string_helper<DType>(os);
      return;
    }
#endif  // MXNET_USE_CUDA
    int dimension = tb_.ndim();
    std::vector<unsigned int> multiples;
    int multiple = 1;
    for (int i = dimension-1; i >= 0; i--) {
      multiple *= tb_.shape_[i];
      multiples.push_back(multiple);
    }
    *os << std::string(dimension, '[');
    *os << tb_.dptr<DType>()[0];
    for (size_t i = 1; i < tb_.shape_.Size(); i++) {
      int n = 0;
      for (auto divisor : multiples) {
        n += (i % divisor == 0);
      }
      if (n) {
        *os << std::string(n, ']') << ", " <<  std::string(n, '[');
      } else  {
        *os << ", ";
      }
      *os << tb_.dptr<DType>()[i];
    }
    *os << std::string(dimension, ']') << std::endl;
    tensor_info_to_string(os);
  }

  /*!
   * \brief output the tensor in a structed format 
   * \tparam DType the data type
   * \tparam StreamType the type of the stream object
   * \param os stream object to output to
   * \param dptr the data pointer
   */
  template<typename DType MSHADOW_DEFAULT_DTYPE, typename StreamType>
  inline void to_string_helper(StreamType* os, const DType* dptr) {
#if MXNET_USE_CUDA
    if (tb_.dev_mask() == gpu::kDevMask) {
      TensorInspector(test::CAccessAsCPU(ctx_, tb_, false)(), ctx_)
          .to_string_helper<DType>(os, dptr);
      return;
    }
#endif  // MXNET_USE_CUDA
    *os << *dptr << std::endl;
    *os << "<" << typeid(*dptr).name() << ">" << std::endl;
  }

  /*!
   * \brief output a part of the tensor in a structed format 
   * \tparam DType the data type
   * \tparam StreamType the type of the stream object
   * \param os stream object to output to
   * \param sub_shape the sub-shape of the desired part of the tensor
   * \param offset the position of the first value of the desired part of the tensor
   */
  template<typename DType MSHADOW_DEFAULT_DTYPE, typename StreamType>
  inline void to_string_helper(StreamType* os, const std::vector<int>& sub_shape, size_t offset) {
#if MXNET_USE_CUDA
    if (tb_.dev_mask() == gpu::kDevMask) {
      TensorInspector(test::CAccessAsCPU(ctx_, tb_, false)(), ctx_)
          .to_string_helper<DType>(os, sub_shape, offset);
      return;
    }
#endif  // MXNET_USE_CUDA
    DType* dptr = tb_.dptr<DType>() + offset;
    if (sub_shape.size() == 0) {
      to_string_helper<DType>(os, dptr);
      return;
    }
    int dimension = sub_shape.size();
    std::vector<int> multiples;
    size_t multiple = 1;
    for (int i = dimension-1; i >= 0; i--) {
      multiple *= sub_shape[i];
      multiples.push_back(multiple);
    }
    std::stringstream ss;
    *os << std::string(dimension, '[');
    *os << dptr[0];
    for (size_t i = 1; i < multiple; i++) {
      int n = 0;
      for (auto divisor : multiples) {
        n += (i % divisor == 0);
      }
      if (n) {
        *os << std::string(n, ']') << ", " <<  std::string(n, '[');
      } else  {
        *os << ", ";
      }
      *os << dptr[i];
    }
    *os << std::string(dimension, ']') << std::endl;
    tensor_info_to_string(os, sub_shape);
  }

  /*!
   * \brief helper function to calculate the sub_shape and offset for the desired part of the tensor,
   * given its coordinates in the original tensor
   * \param pos the coordinates of the desired part of the tensor
   * \param sub_shape the sub-shape of the desired part of the tensor; calculated here
   * \param offset the position of the first value of the desired part of the tensor; calculated here
   */
  inline void print_locator(const std::vector<int>& pos, std::vector<int>* sub_shape,
      size_t* offset) {
    int dimension = tb_.ndim();
    int sub_dim = dimension - pos.size();
    sub_shape->resize(sub_dim);
    int multiple = 1;
    for (int i = pos.size(), j = 0; i < dimension; i++, j++) {
      (*sub_shape)[j] = tb_.shape_[i];
      multiple *= tb_.shape_[i];
    }
    int sum = 0;
    int m = 1;
    for (int i = pos.size()-1; i >= 0; i--) {
      sum += pos[i] * m;
      m *= tb_.shape_[i];
    }
    *offset = sum * multiple;
  }

  /*!
   * \brief parse the coordinate of the desired part of the tensor, given a string that represents that
   * coordinate
   * \param pos the coordinates of the desired part of the tensor, calculated here
   * \param str the string that represents the coordinate
   */
  inline bool parse_position(std::vector<int>* pos, const std::string& str) {
    int dimension = tb_.ndim();
    std::stringstream ss(str);
    int i;
    while (ss >> i) {
      pos->push_back(i);
      if (ss.peek() == ',') {
        ss.ignore();
      }
    }
    if (pos->size() > dimension) {
      return false;
    }
    for (unsigned i = 0; i < pos->size(); i++) {
      if ((*pos)[i] > (tb_.shape_[i]-1)) {
        return false;
      }
    }
    return !pos->empty();
  }

  /*!
   * \brief interactive print the tensor value
   * \tparam DType the data type
   * \param tag the name given to this call
   */
  template<typename DType MSHADOW_DEFAULT_DTYPE>
  inline void interactive_print_helper(std::string tag) {
#if MXNET_USE_CUDA
    if (tb_.dev_mask() == gpu::kDevMask) {
      TensorInspector(test::CAccessAsCPU(ctx_, tb_, false)(), ctx_)
          .interactive_print_helper<DType>(tag);
      return;
    }
#endif  // MXNET_USE_CUDA
    std::lock_guard<std::mutex> lock(InspectorManager::get()->mutex_);
    InspectorManager::get()->interactive_print_tag_counter_[tag] += 1;
    while (!InspectorManager::get()->interactive_print_skip_all_) {
      std::cout << "----------Interactive Print----------" << std::endl;
      if (tag != "") {
        std::cout << "Tag: " << tag << "  Visit: " <<
            InspectorManager::get()->interactive_print_tag_counter_[tag] <<  std::endl;
      }
      tensor_info_to_string(&std::cout);
      std::cout << "Please specify the position, seperated by \",\"" << std::endl;
      std::cout << "\"e\" for the entire tensor, \"d\" to dump value to file," <<
          " \"b\" to break, \"s\" to skip all: ";
      std::string str;
      std::cin >> str;
      if (str == "b") {
        break;
      } else if (str == "e") {
        to_string_helper<DType>(&std::cout);
        continue;
      } else if (str == "s") {
        InspectorManager::get()->interactive_print_skip_all_ = true;
        break;
      } else if (str == "d") {
        while (true) {
          std::cout << "Please enter a tag: ";
          std::cin >> str;
          if (str.find(' ') != std::string::npos) {
            std::cout << "Invalid input. ";
            continue;
          }
          dump_value_helper(str);
          break;
        }
        continue;
      }
      std::vector<int> pos;
      if (parse_position(&pos, str)) {
        std::vector<int> sub_shape;
        size_t offset;
        print_locator(pos, &sub_shape, &offset);
        to_string_helper<DType>(&std::cout, sub_shape, offset);
      } else {
        std::cout << "invalid input" << std::endl;
      }
    }
  }


  /*!
   * \brief build the lambda function, aka the checker, given its type
   * \tparam DType the data type
   * \param ct the type of the checker
   */
  template<typename DType MSHADOW_DEFAULT_DTYPE>
  inline std::function<bool(DType)> build_checker(CheckerType ct) {
    switch (ct) {
      case NegativeChecker:
        return [](DType x) {
              return x < 0;
            };
      case PositiveChecker:
        return [](DType x) {
              return x > 0;
            };
      case ZeroChecker:
        return [](DType x) {
              return x == 0;
            };
      case NaNChecker:
          return [](DType x) {
              return x != x;
            };
      case InfChecker:
        // return float_type_only<DType>([](DType x) {
        //       return x == (DType)1.0 / (DType)0.0 || x == -(DType)1.0 / (DType)0.0;
        //     });
      case PositiveInfChecker:
        return [] (DType x) {
              static_if<std::is_same<DType, float>::value>([x](){return x == get_inf<DType>();}).else_([](){return false;});
              return false;
            };


        // FLOAT_TYPE_ONLY(DType, {
        //   return [](DType x) {
        //         return x == (DType)1.0 / (DType)0.0;
        //       };
        // });
        // break;
        // return float_type_only<DType>([](DType x) {
        //       return x == (DType)1.0 / (DType)0.0;
        //     });
      case NegativeInfChecker:
        // return [] (DType x) {
        //       static_if<std::is_same<DType, float>::value>([x]{return x == -(DType)1.0 / (DType)0.0;}).else_([]{return false;});
        //       return false;
        //     };
      case FiniteChecker:
        // return float_type_only<DType>([](DType x) {
        //       return x != (DType)1.0 / (DType)0.0 && x != -(DType)1.0 / (DType)0.0;
        //     });
      case NormalChecker:
        // return float_type_only<DType>([](DType x) {
        //       return x != (DType)1.0 / (DType)0.0 && x != -(DType)1.0 / (DType)0.0 &&
        //           x == x;
        //     });     
      default:
        return [] (DType x) {
              return false;
            };
    }
    return [] (DType x) {return false;};
  }

  /*!
   * \brief calculate the coordinate of a value in the tensor, given its index
   * \param idx the index of the value in the tensor
   */
  inline std::vector<int> index_to_coordinates(size_t idx) {
    int dimension = tb_.ndim();
    std::vector<int> ret;
    for (int i = dimension-1; i >= 0; i--) {
      ret.push_back(idx % tb_.shape_[i]);
      idx /= tb_.shape_[i];
    }
    std::reverse(ret.begin(), ret.end());
    return ret;
  }

  /*!
   * \brief check/validate the values within the tensor, find the coordinates
   * where the lambda evaluates to true
   * \tparam DType the data type
   * \param ret a vector of coordinates which itself is a vector of int; calculated here
   * \param checker the lambda function to check each value of within the tensor
   * \param interactive wherether to allow the user to interactively check the coordinates
   * \param tag the name given to this call
   */
  template<typename DType MSHADOW_DEFAULT_DTYPE>
  inline void check_value_helper(std::vector<std::vector<int>>* ret,
      const std::function<bool(DType)>& checker, bool interactive, std::string tag) {
#if MXNET_USE_CUDA
    if (tb_.dev_mask() == gpu::kDevMask) {
      return TensorInspector(test::CAccessAsCPU(ctx_, tb_, false)(), ctx_)
          .check_value_helper<DType>(ret, checker, interactive, tag);
    }
#endif  // MXNET_USE_CUDA
    int count = 0;
    std::stringstream ss;
    ss << "[";
    bool first_pass = true;
    for (size_t i = 0; i < tb_.shape_.Size(); i++) {
      if (checker(tb_.dptr<DType>()[i])) {
        count += 1;
        if (!first_pass) {
          ss  << ", ";
        }
        first_pass = false;
        std::vector<int> coords = index_to_coordinates(i);
        ss << "(" << coords[0];
        for (size_t i = 1; i < coords.size(); i++) {
          ss << ", " << coords[i];
        }
        ss << ")";
        ret->push_back(coords);
      }
    }
    ss << "]" << std::endl;
    if (interactive) {
      std::lock_guard<std::mutex> lock(InspectorManager::get()->mutex_);
       InspectorManager::get()->check_value_tag_counter_[tag] += 1;
      while (!InspectorManager::get()->check_value_skip_all_) {
        std::cout << "----------Value Check----------" << std::endl;
        if (tag != "") {
          std::cout << "Tag: " << tag << "  Visit: " <<
              InspectorManager::get()->check_value_tag_counter_[tag] <<  std::endl;
        }
        std::cout << count << " value(s) found." << std::endl;
        std::cout << "\"p\" to print the coordinates, \"b\" to break, \"s\" to skip all: ";
        std::string str;
        std::cin >> str;
        if (str == "b") {
          break;
        } else if (str == "p") {
          std::cout << ss.str() << std::endl;
        } else if (str == "s") {
          InspectorManager::get()->check_value_skip_all_ = true;
        }
      }
    }
  }

  /*!
   * \brief infer the python type, given the c++ type
   * \tparam ti the type info 
   */
  inline char infer_type(const std::type_info& ti) {
    if (ti == typeid(float)) return 'f';
    else if (ti == typeid(double)) return 'f';
    else if (ti == typeid(mshadow::half::half_t) ) return 'f';
    else if (ti == typeid(uint8_t)) return 'u';
    else if (ti == typeid(int32_t)) return 'i';
    else if (ti == typeid(int64_t)) return 'i';
    else
      return '?';
  }

  /*!
   * \brief check if the host machine is big or small endian
   */
  inline char endian_test() {
    int x = 1;
    return (reinterpret_cast<char*>(&x)[0]) ? '<' : '>';
  }

  /*!
   * \brief dump the value of the tensor to a file with name "tag_[visit count].npy" in npy format
   * \tparam DType the data type
   * \param tag the name given to this call
   */
  template<typename DType MSHADOW_DEFAULT_DTYPE>
  inline void dump_value_helper(const std::string& tag) {
#if MXNET_USE_CUDA
    if (tb_.dev_mask() == gpu::kDevMask) {
      TensorInspector(test::CAccessAsCPU(ctx_, tb_, false)(), ctx_)
          .dump_value_helper<DType>(tag);
      return;
    }
#endif  // MXNET_USE_CUDA
    std::string dict;
    dict += "{'descr':'";
    dict += endian_test();
    dict += infer_type(typeid(DType));
    dict += std::to_string(sizeof(DType));
    dict += "','fortran_order':False,'shape':(";
    dict += std::to_string(tb_.shape_[0]);
    for (int i = 1; i < tb_.ndim(); i++) {
      dict += ',';
      dict += std::to_string(tb_.shape_[i]);
    }
    if (tb_.ndim() == 1) {
       dict += ",";
    }
    dict += ")} ";
    int padding_size = 64 - ((10 + dict.size()) % 64);
    dict += std::string(padding_size, ' ');
    dict[dict.size()-1] = '\n';
    std::string header;
    header += static_cast<char>(0x93);
    header += "NUMPY";
    header += static_cast<char>(0x01);
    header += static_cast<char>(0x00);
    header += static_cast<char>((uint16_t)dict.size() & 0x00ff);
    header += static_cast<char>(((uint16_t)dict.size() >> 8) & 0x00ff);
    header += dict;
    InspectorManager::get()->dump_value_tag_counter_[tag] += 1;
    int visit = InspectorManager::get()->dump_value_tag_counter_[tag];
    std::ofstream file(tag + "_" + std::to_string(visit) + ".npy",
        std::ios::out | std::ios::binary);
    file.write(header.c_str(), header.size());
    file.write(reinterpret_cast<char*>(tb_.dptr<DType>()), sizeof(DType) * tb_.shape_.Size());
    file.close();
  }

  /* !\brief the tensor blob */
  const TBlob tb_;
  /* !\brief the run context of the tensor */
  const RunContext& ctx_;

 public:
   /*!
   * \brief Construct from Tensor object
   * \tparam Device the device the tensor resides in
   * \tparam dimension the dimension of the tensor
   * \tparam DType the data type
   * \param ts the source tensor object
   * \param ctx the run context of the tensor
   */
  template<typename Device, int dimension,
      typename DType MSHADOW_DEFAULT_DTYPE>
  TensorInspector(const Tensor<Device, dimension, DType>& ts, const RunContext& ctx):
      tb_(ts), ctx_(ctx) {}

  /*!
   * \brief Construct from TBlob object
   * \param tb the source tblob object
   * \param ctx the run context of the tensor
   */
  TensorInspector(const TBlob& tb, const RunContext& ctx):
      tb_(tb), ctx_(ctx) {}

  /*!
   * \brief Construct from NDArray object. Currently this only works with kDefaultStorage
   * \param arr the source ndarray object
   * \param ctx the run context of the tensor
   */
  TensorInspector(const NDArray& arr, const RunContext& ctx):
      tb_(arr.data()), ctx_(ctx) {}

  /*!
   * \brief print the tensor to std::cout
   */
  inline void print_string() {
    std::cout << to_string() << std::endl;
  }

  /*!
   * \brief return a string which contains the values and other info of the tensor
   */
  inline std::string to_string() {
    std::stringstream ss;
    MSHADOW_TYPE_SWITCH(tb_.type_flag_, DType, {
      to_string_helper<DType>(&ss);
    });
    return ss.str();
  }

  /*!
   * \brief interactive print the tensor value
   * \param tag the name given to this call
   */
  inline void interactive_print(std::string tag = "") {
    MSHADOW_TYPE_SWITCH(tb_.type_flag_, DType, {
      interactive_print_helper<DType>(tag);
    });
  }

  /*!
   * \brief check/validate the values within the tensor, return the coordinates
   * where the lambda evaluates to true
   * \tparam ValueChecker the type of the lambda
   * \param checker the lambda function to check each value of within the tensor
   * \param interactive wherether to allow the user to interactively check the coordinates
   * \param tag the name given to this call
   */
  template<typename ValueChecker>
  inline std::vector<std::vector<int>> check_value(const ValueChecker& checker,
      bool interactive = false, std::string tag = "") {
    std::vector<std::vector<int>> ret;
    MSHADOW_TYPE_SWITCH(tb_.type_flag_, DType, {
      check_value_helper<DType>(&ret, checker, ret, interactive, tag);
    });
    return ret;
  }

  /*!
   * \brief check/validate the values within the tensor, return the coordinates
   * where the lambda evaluates to true
   * \param ct the type of the checker
   * \param interactive wherether to allow the user to interactively check the coordinates
   * \param tag the name given to this call
   */
  inline std::vector<std::vector<int>> check_value(CheckerType ct, bool interactive = false,
      std::string tag = "") {
    std::vector<std::vector<int>> ret;
    MSHADOW_TYPE_SWITCH(tb_.type_flag_, DType, {
      check_value_helper<DType>(&ret, build_checker<DType>(ct), interactive, tag);
    });
    return ret;
  }

  /*!
   * \brief dump the value of the tensor to a file with name "tag_[visit count].npy" in npy format
   * \param tag the name given to this call
   */
  inline void dump_value(std::string tag) {
    MSHADOW_TYPE_SWITCH(tb_.type_flag_, DType, {
      dump_value_helper<DType>(tag);
    });
  }
};

}  // namespace mxnet

#endif  // MXNET_COMMON_TENSOR_INSPECTOR_H_
