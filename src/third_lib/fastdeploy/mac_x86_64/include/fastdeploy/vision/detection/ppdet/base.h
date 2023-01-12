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
#include "fastdeploy/vision/detection/ppdet/preprocessor.h"
#include "fastdeploy/vision/detection/ppdet/postprocessor.h"
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {
namespace vision {
/** \brief All object detection model APIs are defined inside this namespace
 *
 */
namespace detection {

/*! @brief Base model object used when to load a model exported by PaddleDetection
 */
class FASTDEPLOY_DECL PPDetBase : public FastDeployModel {
 public:
  /** \brief Set path of model file and configuration file, and the configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g ppyoloe/model.pdmodel
   * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams, if the model format is ONNX, this parameter will be ignored
   * \param[in] config_file Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
   * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
   * \param[in] model_format Model format of the loaded model, default is Paddle format
   */
  PPDetBase(const std::string& model_file, const std::string& params_file,
          const std::string& config_file,
          const RuntimeOption& custom_option = RuntimeOption(),
          const ModelFormat& model_format = ModelFormat::PADDLE);

  /** \brief Clone a new PaddleDetModel with less memory usage when multiple instances of the same model are created
   *
   * \return new PaddleDetModel* type unique pointer
   */
  virtual std::unique_ptr<PPDetBase> Clone() const;

  /// Get model's name
  virtual std::string ModelName() const { return "PaddleDetection/BaseModel"; }

  /** \brief DEPRECATED Predict the detection result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
   * \param[in] result The output detection result
   * \return true if the prediction successed, otherwise false
   */
  virtual bool Predict(cv::Mat* im, DetectionResult* result);

  /** \brief Predict the detection result for an input image
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
   * \param[in] result The output detection result
   * \return true if the prediction successed, otherwise false
   */
  virtual bool Predict(const cv::Mat& im, DetectionResult* result);

  /** \brief Predict the detection result for an input image list
   * \param[in] im The input image list, all the elements come from cv::imread(), is a 3-D array with layout HWC, BGR format
   * \param[in] results The output detection result list
   * \return true if the prediction successed, otherwise false
   */
  virtual bool BatchPredict(const std::vector<cv::Mat>& imgs,
                            std::vector<DetectionResult>* results);

  PaddleDetPreprocessor& GetPreprocessor() {
    return preprocessor_;
  }

  PaddleDetPostprocessor& GetPostprocessor() {
    return postprocessor_;
  }

 protected:
  virtual bool Initialize();
  PaddleDetPreprocessor preprocessor_;
  PaddleDetPostprocessor postprocessor_;
};

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
