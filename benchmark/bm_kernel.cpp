#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

class BmKernel {
 public:
  BmKernel(int rows, int rank, const std::int32_t* indptr,
           const std::int32_t* indices, const double* values, int nnz)
      : rows_(rows),
        rank_(rank),
        indptr_(indptr, indptr + rows + 1),
        indices_(indices, indices + nnz),
        values_(values, values + nnz),
        normalized_(static_cast<std::size_t>(rows) * rank),
        product_(static_cast<std::size_t>(rows) * rank),
        inverse_norms_(rows) {}

  double evaluate(const double* z, double* gradient) {
    for (int row = 0; row < rows_; ++row) {
      const std::size_t offset = static_cast<std::size_t>(row) * rank_;
      double squared_norm = 0.0;
      for (int column = 0; column < rank_; ++column) {
        squared_norm += z[offset + column] * z[offset + column];
      }
      const double norm = std::max(std::sqrt(squared_norm), 1e-12);
      inverse_norms_[row] = 1.0 / norm;
      for (int column = 0; column < rank_; ++column) {
        normalized_[offset + column] = z[offset + column] / norm;
        product_[offset + column] = 0.0;
      }
    }

    for (int row = 0; row < rows_; ++row) {
      const std::size_t output_offset = static_cast<std::size_t>(row) * rank_;
      for (int entry = indptr_[row]; entry < indptr_[row + 1]; ++entry) {
        const std::size_t input_offset =
            static_cast<std::size_t>(indices_[entry]) * rank_;
        const double weight = values_[entry];
        for (int column = 0; column < rank_; ++column) {
          product_[output_offset + column] +=
              weight * normalized_[input_offset + column];
        }
      }
    }

    double value = 0.0;
    for (std::size_t index = 0; index < normalized_.size(); ++index) {
      value += normalized_[index] * product_[index];
    }

    for (int row = 0; row < rows_; ++row) {
      const std::size_t offset = static_cast<std::size_t>(row) * rank_;
      double radial = 0.0;
      for (int column = 0; column < rank_; ++column) {
        radial += 2.0 * product_[offset + column] *
                  normalized_[offset + column];
      }
      const double inverse_norm = inverse_norms_[row];
      for (int column = 0; column < rank_; ++column) {
        gradient[offset + column] =
            (2.0 * product_[offset + column] -
             radial * normalized_[offset + column]) *
            inverse_norm;
      }
    }
    return value;
  }

 private:
  int rows_;
  int rank_;
  std::vector<std::int32_t> indptr_;
  std::vector<std::int32_t> indices_;
  std::vector<double> values_;
  std::vector<double> normalized_;
  std::vector<double> product_;
  std::vector<double> inverse_norms_;
};

extern "C" void* bm_kernel_create(int rows, int rank,
                                   const std::int32_t* indptr,
                                   const std::int32_t* indices,
                                   const double* values, int nnz) {
  try {
    return new BmKernel(rows, rank, indptr, indices, values, nnz);
  } catch (...) {
    return nullptr;
  }
}

extern "C" void bm_kernel_destroy(void* handle) {
  delete static_cast<BmKernel*>(handle);
}

extern "C" int bm_kernel_evaluate(void* handle, const double* z,
                                  double* gradient, double* value) {
  if (!handle || !z || !gradient || !value) return -1;
  try {
    *value = static_cast<BmKernel*>(handle)->evaluate(z, gradient);
    return 0;
  } catch (...) {
    return -2;
  }
}
