#ifndef MXNET_COMMON_TENSOR_INSPECTOR_H_
#define MXNET_COMMON_TENSOR_INSPECTOR_H_

#include <algorithm> 
#include "../../3rdparty/mshadow/mshadow/base.h"
#include "../../tests/cpp/include/test_util.h"
namespace mxnet{

class InspectorManager {
 public:
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

  bool skip_all_ = false;
  std::unordered_map<std::string, int> tag_counter_;
};

enum CheckerType {
  NegativeChecker,
  PositiveChecker
};

template<typename DType MSHADOW_DEFAULT_DTYPE>
std::function<bool(DType)> build_checker(CheckerType ct){
  switch (ct) {
    case NegativeChecker:
      return [] (DType x) {
          return x < 0;
        };
      break;
    case PositiveChecker:
      return [] (DType x) {
          return x < 0;
        };
      break;
  }
  return [] (DType x) {return true;};
};


class TensorInspector {
 private:
  const TBlob tb_;

  template<typename DType MSHADOW_DEFAULT_DTYPE, typename StreamType>
  inline void tensor_info_to_string(StreamType& os) {
    int dimension = tb_.ndim();
    os << "<" << typeid(tb_.dptr<DType>()[0]).name() << " Tensor ";
    os << tb_.shape_[0];
    for (int i = 1; i < dimension; i++) {
      os << 'x' << tb_.shape_[i];
    }
    os << ">" << std::endl;
  }

  template<typename DType MSHADOW_DEFAULT_DTYPE, typename StreamType>
  inline void tensor_info_to_string(StreamType& os, const std::vector<int>& shape) {
    int dimension = shape.size();
    os << "<" << typeid(tb_.dptr<DType>()[0]).name() << " Tensor ";
    os << shape[0];
    for (int i = 1; i < dimension; i++) {
      os << 'x' << shape[i];
    }
    os << ">" << std::endl;
  }

  template<typename DType MSHADOW_DEFAULT_DTYPE, typename StreamType>
  inline void to_string_helper(const RunContext& ctx, StreamType& os) { 
#if MXNET_USE_CUDA
    if (tb_.dev_mask() == gpu::kDevMask) {
      TensorInspector(test::CAccessAsCPU(ctx, tb_, false)()).to_string_helper<DType>(ctx, os);
      return;
    }
#endif // MXNET_USE_CUDA
    int dimension = tb_.ndim();
    std::vector<unsigned int> multiples;
    int multiple = 1;
    for (int i = dimension-1; i >= 0; i--) {
      multiple *= tb_.shape_[i];
      multiples.push_back(multiple);
    }
    os << std::string(dimension, '[');
    os << tb_.dptr<DType>()[0];
    for (size_t i = 1; i < tb_.shape_.Size(); i++) {
      int n = 0;
      for (auto divisor : multiples) {
        n += (i % divisor == 0);
      }
      if (n) {
        os << std::string(n, ']') << ", " <<  std::string(n, '[');
      } else  {
        os << ", ";
      }
      os << tb_.dptr<DType>()[i];
    }
    os << std::string(dimension, ']') << std::endl;
    tensor_info_to_string(os);
  }

  template<typename DType MSHADOW_DEFAULT_DTYPE, typename StreamType>
  inline void to_string_helper(const RunContext& ctx, StreamType& os, const DType* dptr) {
#if MXNET_USE_CUDA
    if (tb_.dev_mask() == gpu::kDevMask) {
      TensorInspector(test::CAccessAsCPU(ctx, tb_, false)()).to_string_helper<DType>(ctx, os, dptr);
      return;
    }
#endif // MXNET_USE_CUDA
    os << *dptr << std::endl;
    os << "<" << typeid(*dptr).name() << ">" << std::endl;
  }


  template<typename DType MSHADOW_DEFAULT_DTYPE, typename StreamType>
  inline void to_string_helper(const RunContext& ctx, StreamType& os, const std::vector<int>& sub_shape, size_t offset) {
#if MXNET_USE_CUDA
    if (tb_.dev_mask() == gpu::kDevMask) {
      TensorInspector(test::CAccessAsCPU(ctx, tb_, false)()).to_string_helper<DType>(ctx, os, sub_shape, offset);
      return;
    }
#endif // MXNET_USE_CUDA
    DType* dptr = tb_.dptr<DType>() + offset;
    if (sub_shape.size() == 0) {
      to_string_helper<DType>(ctx, os, dptr);
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
    os << std::string(dimension, '[');
    os << dptr[0];
    for (size_t i = 1; i < multiple; i++) {
      int n = 0;
      for (auto divisor : multiples) {
        n += (i % divisor == 0);
      }
      if (n) {
        os << std::string(n, ']') << ", " <<  std::string(n, '[');
      } else  {
        os << ", ";
      }
      os << dptr[i];
    }
    os << std::string(dimension, ']') << std::endl;
    tensor_info_to_string(os, sub_shape);
    os << "offset" << offset <<std::endl;
  }

  inline void print_locator(const std::vector<int>& pos, std::vector<int>& sub_shape, size_t& offset) {
    int dimension = tb_.ndim();
    int sub_dim = dimension - pos.size();
    sub_shape.resize(sub_dim);
    int multiple = 1;
    for (int i = pos.size(), j = 0; i < dimension; i++, j++) {
      sub_shape[j] = tb_.shape_[i];
      multiple *= tb_.shape_[i];
    }
    int sum = 0;
    int m = 1;
    for (int i = pos.size()-1; i >= 0; i--) {
      sum += pos[i] * m;
      m *= tb_.shape_[i];
    }
    offset = sum * multiple;
  }

  inline bool parse_position(std::vector<int>& pos, std::string str) {
    int dimension = tb_.ndim();
    std::stringstream ss(str);
    int i;
    while (ss >> i) {
      pos.push_back(i);
      if (ss.peek() == ',') {
        ss.ignore();
      }
    }
    if (pos.size() > dimension) {
      return false;
    }
    for (unsigned i = 0; i < pos.size(); i++) {
      if (pos[i] > (tb_.shape_[i]-1)) {
        return false;
      }
    }
    return !pos.empty();
  }

  template<typename DType MSHADOW_DEFAULT_DTYPE>
  inline void interactive_print_helper(const RunContext& ctx, std::string tag) {
#if MXNET_USE_CUDA
    if (tb_.dev_mask() == gpu::kDevMask) {
      TensorInspector(test::CAccessAsCPU(ctx, tb_, false)()).interactive_print_helper<DType>(ctx, tag);
      return;
    }
#endif // MXNET_USE_CUDA
    InspectorManager::get()->tag_counter_[tag] += 1;
    while (!InspectorManager::get()->skip_all_) {
      std::cout << "----------Interactive Print----------" << std::endl;
      if (tag != "") {
        std::cout << "Tag: " << tag << "  Visit: " << InspectorManager::get()->tag_counter_[tag] <<  std::endl;
      }
      tensor_info_to_string(std::cout);
      std::cout << "Please specify the position, seperated by \",\"" << std::endl
          << "\"e\" for the entire tensor ,\"b\" to break, \"s\" to skip all: " << std::endl;
      std::string str;
      std::cin >> str;
      if (str == "b") {
        break;
      } else if (str == "e") {
        to_string_helper<DType>(ctx, std::cout);
        continue;
      } else if (str == "s") {
        InspectorManager::get()->skip_all_ = true;
        break;
      }
      std::vector<int> pos;
      if (parse_position(pos, str)) {
        std::vector<int> sub_shape;
        size_t offset;
        print_locator(pos, sub_shape, offset);
        to_string_helper<DType>(ctx, std::cout, sub_shape, offset);
      } else {
        std::cout << "invalid input" << std::endl;
      }
    }
  }


 public:
  template<typename Device, int dimension,
      typename DType MSHADOW_DEFAULT_DTYPE>
  TensorInspector(const Tensor<Device, dimension, DType>& ts) : tb_(ts) {}

  TensorInspector(const TBlob& tb) : tb_(tb) {}

  // Only works for kDefaultStorage
  TensorInspector(const NDArray& arr) : tb_(arr.data()){}

  inline void print_string(const RunContext& ctx) {
    std::cout << to_string(ctx) << std::endl;
  }

  inline std::string to_string(const RunContext& ctx) {
    std::stringstream ss;
    MSHADOW_TYPE_SWITCH(tb_.type_flag_, DType, {
      to_string_helper<DType>(ctx, ss);
    });
    return ss.str();
  }

  inline void interactive_print(const RunContext& ctx, std::string tag = "") {
    MSHADOW_TYPE_SWITCH(tb_.type_flag_, DType, {
      interactive_print_helper<DType>(ctx, tag);
    });
  }
};


// template<typename Device, int dimension,
//     typename DType MSHADOW_DEFAULT_DTYPE>
// class TensorInspector2 {
//  private:
//   const Tensor<Device, dimension, DType> ts_;
//   static const int  kSubdim = dimension - 1;


//  public:
//   TensorInspector(const Tensor<Device, dimension, DType>& ts) : ts_(ts){}

//   TensorInspector<Device, kSubdim, DType> operator[](index_t idx) const {
//     return TensorInspector<Device, kSubdim, DType>(ts_[idx]);
//   }

//   std::vector<int> index_to_coordinates(unsigned idx){
//     std::vector<int> ret;
//     for (int i = dimension-1; i >= 0; i--) {
//       ret.push_back(idx % ts_.shape_[i]);
//       idx /= ts_.shape_[i];
//     }
//     std::reverse(ret.begin(), ret.end());
//     return ret;
//   }

//   // std::function<bool(DType)> negative_checker = [] (DType x)
//   // {
//   //     return x < 0;
//   // };

//   template<typename ValueChecker>
//   std::vector<std::vector<int>> check_value(ValueChecker value_checker) {
//     std::vector<std::vector<int>> ret;
//     std::stringstream ss;
//     ss << "[";
//     bool first_pass = true;
//     for (int i = 0; i < ts_.MSize(); i++) {
//       if (value_checker(*(ts_.dptr_ + i))) {
//         if (!first_pass) {
//           ss << ", ";
//         }
//         first_pass = false;
//         std::vector<int> coords = index_to_coordinates(i);
//         ss << "(" << coords[0];
//         for (int i = 1; i < coords.size(); i++) {
//           ss << ", " << coords[i];
//         }
//         ss << ")";
//         ret.push_back(coords);
//       }
//     }
//     ss << "]" << std::endl;
//     std::cout << ss.str();
//     return ret;
//   }

//   bool parse_position(std::vector<int>& pos, std::string str) {
//     std::stringstream ss(str);
//     int i;
//     while (ss >> i) {
//       pos.push_back(i);
//       if (ss.peek() == ',') {
//         ss.ignore();
//       }
//     }
//     if (pos.size() > dimension) {
//       return false;
//     }
//     for (unsigned i = 0; i < pos.size(); i++) {
//       if (pos[i] > (ts_.shape_[i]-1)) {
//         return false;
//       }
//     }
//     return !pos.empty();
//   }

  // void print_helper(std::vector<int>& pos) {
  //   int sub_dim = dimension - pos.size();
  //   std::vector<int> sub_shape(sub_dim);
  //   int multiple = 1;
  //   for (int i = pos.size(), j = 0; i < dimension; i++, j++) {
  //     sub_shape[j] = ts_.shape_[i];
  //     multiple *= ts_.shape_[i];
  //   }
  //   int sum = 0;
  //   int m = 1;
  //   for (int i = pos.size()-1; i >= 0; i--) {
  //     sum += pos[i] * m;
  //     m *= ts_.shape_[i];
  //   }
  //   int offset = sum * multiple;
  //   to_string_helper<DType>(ts_.dptr_ + offset, sub_shape);
  // }

//   void interactive_print(std::string tag = "") {
//     InspectorManager::get()->tag_counter_[tag] += 1;
//     while (!InspectorManager::get()->skip_all_) {
//       std::cout << "----------Interactive Print----------" << std::endl;
//       if (tag != "") {
//         std::cout << "Tag: " << tag << "  Visit: " << InspectorManager::get()->tag_counter_[tag] <<  std::endl;
//       }
//       std::cout << "<" << typeid(*ts_.dptr_).name() << " Tensor ";
//       std::cout << ts_.shape_[0];
//       for (int i = 1; i < dimension; i++) {
//         std::cout << 'x' << ts_.shape_[i];
//       }
//     std::cout << ">" << std::endl;
//       std::cout << "Please specify the position, seperated by \",\"" << std::endl
//           << "\"e\" for the entire tensor ,\"b\" to break, \"s\" to skip all: " << std::endl;
//       std::string str;
//       std::cin >> str;
//       if (str == "b") {
//         break;
//       } else if (str == "e") {
//         to_string();
//         continue;
//       } else if (str == "s") {
//         InspectorManager::get()->skip_all_ = true;
//         break;
//       }
//       std::vector<int> pos;
//       if (parse_position(pos, str)) {
//         print_helper(pos);
//       } else {
//         std::cout << "invalid input" << std::endl;
//       }
//     }
//   }

//   std::string to_string() {
//     std::vector<int> multiples;
//     int multiple = 1;
//     for (int i = dimension-1; i >= 0; i--) {
//       multiple *= ts_.shape_[i];
//       multiples.push_back(multiple);
//     }
//     std::stringstream ss;
//     ss << std::string(dimension, '[');
//     ss << *(DType*)ts_.dptr_;
//     for (int i = 1; i < multiple; i++) {
//       int n = 0;
//       for (int divisor : multiples) {
//         n += (i % divisor == 0);
//       }
//       if (n) {
//         ss << std::string(n, ']') << ", " <<  std::string(n, '[');
//       } else  {
//         ss << ", ";
//       }
//       ss << *(ts_.dptr_ + i);
//     }
//     ss << std::string(dimension, ']') << std::endl
//     ;
//     ss << "<" << typeid(*ts_.dptr_).name() << " Tensor ";
//     ss << ts_.shape_[0];
//     for (int i = 1; i < dimension; i++) {
//       ss << 'x' << ts_.shape_[i];
//     }
//     ss << ">" << std::endl;
//     std::cout << ss.str();
//     return ss.str();
//   }
// };


}
#endif  // MXNET_COMMON_TENSOR_INSPECTOR_H_
