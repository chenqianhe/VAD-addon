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
#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {

namespace vision {

namespace matting {
/*! @brief MODNet model object used when to load a MODNet model exported by MODNet.
 */
class FASTDEPLOY_DECL MODNet : public FastDeployModel {
 public:
  /** \brief  Set path of model file and the configuration of runtime.
   *
   * \param[in] model_file Path of model file, e.g ./modnet.onnx
   * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams, if the model format is ONNX, this parameter will be ignored
   * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in "valid_cpu_backends"
   * \param[in] model_format Model format of the loaded model, default is ONNX format
   */
  MODNet(const std::string& model_file, const std::string& params_file = "",
         const RuntimeOption& custom_option = RuntimeOption(),
         const ModelFormat& model_format = ModelFormat::ONNX);

  std::string ModelName() const { return "matting/MODNet"; }

  /*! @brief
  Argument for image preprocessing step, tuple of (width, height), decide the target size after resize, default (256, 256)
  */
  std::vector<int> size;
  /*! @brief
  Argument for image preprocessing step, parameters for normalization, size should be the the same as channels, default alpha = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f}
  */
  std::vector<float> alpha;
  /*! @brief
  Argument for image preprocessing step, parameters for normalization, size should be the the same as channels, default beta = {-1.f, -1.f, -1.f}
  */
  std::vector<float> beta;
  /*! @brief
  Argument for image preprocessing step, whether to swap the B and R channel, such as BGR->RGB, default true.
  */
  bool swap_rb;
  /** \brief Predict the matting result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
   * \param[in] result The output matting result will be writen to this structure
   * \return true if the prediction successed, otherwise false
   */
  bool Predict(cv::Mat* im, MattingResult* result);

 private:
  bool Initialize();

  bool Preprocess(Mat* mat, FDTensor* output,
                  std::map<std::string, std::array<int, 2>>* im_info);

  bool Postprocess(std::vector<FDTensor>& infer_result, MattingResult* result,
                   const std::map<std::string, std::array<int, 2>>& im_info);
};

}  // namespace matting
}  // namespace vision
}  // namespace fastdeploy
