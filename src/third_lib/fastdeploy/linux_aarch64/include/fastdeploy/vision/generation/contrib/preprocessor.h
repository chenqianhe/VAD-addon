// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/function/functions.h"

namespace fastdeploy {
namespace vision {

namespace generation {
/*! @brief Preprocessor object for AnimeGAN serials model.
 */
class FASTDEPLOY_DECL AnimeGANPreprocessor {
 public:
  /** \brief Create a preprocessor instance for AnimeGAN serials model
   */
  AnimeGANPreprocessor() {}

  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] images The input image data list, all the elements are returned wrapped by FDMat.
   * \param[in] output The output tensors which will feed in runtime
   * \return true if the preprocess successed, otherwise false
   */
  bool Run(std::vector<Mat>& images, std::vector<FDTensor>* output);
};

}  // namespace generation
}  // namespace vision
}  // namespace fastdeploy
