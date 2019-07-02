#ifndef MXNET_COMMON_TENSOR_INSPECTOR_H_
#define MXNET_COMMON_TENSOR_INSPECTOR_H_

#include <algorithm> 
#include "../../3rdparty/mshadow/mshadow/tensor.h"

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

template<typename DType MSHADOW_DEFAULT_DTYPE>
std::string to_string_helper(DType* dptr) {
  std::stringstream ss;
  ss << *dptr << std::endl;
  ss << "<" << typeid(*dptr).name() << ">" << std::endl;
  std::cout << ss.str();
  return ss.str();
}

template<typename DType MSHADOW_DEFAULT_DTYPE>
std::string to_string_helper(DType* dptr, std::vector<int> shape) {
  if (shape.size() == 0) {
    return to_string_helper(dptr);
  }
  int dimension = shape.size();

  std::vector<int> multiples;
  int multiple = 1;
  for (int i = dimension-1; i >= 0; i--) {
    multiple *= shape[i];
    multiples.push_back(multiple);
  }
  std::stringstream ss;
  ss << std::string(dimension, '[');
  ss << *dptr;
  for (int i = 1; i < multiple; i++) {
    int n = 0;
    for (int divisor : multiples) {
      n += (i % divisor == 0);
    }
    if (n) {
      ss << std::string(n, ']') << ", " <<  std::string(n, '[');
    } else  {
      ss << ", ";
    }
    ss << *(dptr + i);
  }
  ss << std::string(dimension, ']') << std::endl
  ;
  ss << "<" << typeid(*dptr).name() <<" Tensor ";
  ss << shape[0];
  for (int i = 1; i < dimension; i++) {
    ss << 'x' << shape[i];
  }
  ss << ">" << std::endl;
  std::cout << ss.str();
  return ss.str();
}

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




template<typename Device, int dimension,
    typename DType MSHADOW_DEFAULT_DTYPE>
class TensorInspector {
 private:
  const Tensor<Device, dimension, DType> ts_;
  static const int  kSubdim = dimension - 1;

  


 public:
  TensorInspector(const Tensor<Device, dimension, DType>& ts) : ts_(ts){}

  TensorInspector<Device, kSubdim, DType> operator[](index_t idx) const {
    return TensorInspector<Device, kSubdim, DType>(ts_[idx]);
  }

  std::vector<int> index_to_coordinates(unsigned idx){
    std::vector<int> ret;
    for (int i = dimension-1; i >= 0; i--) {
      ret.push_back(idx % ts_.shape_[i]);
      idx /= ts_.shape_[i];
    }
    std::reverse(ret.begin(), ret.end());
    return ret;
  }

  // std::function<bool(DType)> negative_checker = [] (DType x)
  // {
  //     return x < 0;
  // };

  template<typename ValueChecker>
  std::vector<std::vector<int>> check_value(ValueChecker value_checker) {
    std::vector<std::vector<int>> ret;
    std::stringstream ss;
    ss << "[";
    bool first_pass = true;
    for (int i = 0; i < ts_.MSize(); i++) {
      if (value_checker(*(ts_.dptr_ + i))) {
        if (!first_pass) {
          ss << ", ";
        }
        first_pass = false;
        std::vector<int> coords = index_to_coordinates(i);
        ss << "(" << coords[0];
        for (int i = 1; i < coords.size(); i++) {
          ss << ", " << coords[i];
        }
        ss << ")";
        ret.push_back(coords);
      }
    }
    ss << "]" << std::endl;
    std::cout << ss.str();
    return ret;
  }

  bool parse_position(std::vector<int>& pos, std::string str) {
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
      if (pos[i] > (ts_.shape_[i]-1)) {
        return false;
      }
    }
    return !pos.empty();
  }

  void print_helper(std::vector<int>& pos) {
    int sub_dim = dimension - pos.size();
    std::vector<int> sub_shape(sub_dim);
    int multiple = 1;
    for (int i = pos.size(), j = 0; i < dimension; i++, j++) {
      sub_shape[j] = ts_.shape_[i];
      multiple *= ts_.shape_[i];
    }
    int sum = 0;
    int m = 1;
    for (int i = pos.size()-1; i >= 0; i--) {
      sum += pos[i] * m;
      m *= ts_.shape_[i];
    }
    int offset = sum * multiple;
    to_string_helper<DType>(ts_.dptr_ + offset, sub_shape);
  }

  void interactive_print(std::string tag = "") {
    InspectorManager::get()->tag_counter_[tag] += 1;
    while (!InspectorManager::get()->skip_all_) {
      std::cout << "----------Interactive Print----------" << std::endl;
      if (tag != "") {
        std::cout << "Tag: " << tag << "  Visit: " << InspectorManager::get()->tag_counter_[tag] <<  std::endl;
      }
      std::cout << "<" << typeid(*ts_.dptr_).name() << " Tensor ";
      std::cout << ts_.shape_[0];
      for (int i = 1; i < dimension; i++) {
        std::cout << 'x' << ts_.shape_[i];
      }
    std::cout << ">" << std::endl;
      std::cout << "Please specify the position, seperated by \",\"" << std::endl
          << "\"e\" for the entire tensor ,\"b\" to break, \"s\" to skip all: " << std::endl;
      std::string str;
      std::cin >> str;
      if (str == "b") {
        break;
      } else if (str == "e") {
        to_string();
        continue;
      } else if (str == "s") {
        InspectorManager::get()->skip_all_ = true;
        break;
      }
      std::vector<int> pos;
      if (parse_position(pos, str)) {
        print_helper(pos);
      } else {
        std::cout << "invalid input" << std::endl;
      }
    }
  }

  std::string to_string() {
    std::vector<int> multiples;
    int multiple = 1;
    for (int i = dimension-1; i >= 0; i--) {
      multiple *= ts_.shape_[i];
      multiples.push_back(multiple);
    }
    std::stringstream ss;
    ss << std::string(dimension, '[');
    ss << *(DType*)ts_.dptr_;
    for (int i = 1; i < multiple; i++) {
      int n = 0;
      for (int divisor : multiples) {
        n += (i % divisor == 0);
      }
      if (n) {
        ss << std::string(n, ']') << ", " <<  std::string(n, '[');
      } else  {
        ss << ", ";
      }
      ss << *(ts_.dptr_ + i);
    }
    ss << std::string(dimension, ']') << std::endl
    ;
    ss << "<" << typeid(*ts_.dptr_).name() << " Tensor ";
    ss << ts_.shape_[0];
    for (int i = 1; i < dimension; i++) {
      ss << 'x' << ts_.shape_[i];
    }
    ss << ">" << std::endl;
    std::cout << ss.str();
    return ss.str();
  }



};


#endif  // MXNET_COMMON_TENSOR_INSPECTOR_H_