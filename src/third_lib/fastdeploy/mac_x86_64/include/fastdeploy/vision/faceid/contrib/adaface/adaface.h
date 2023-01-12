﻿// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.  //NOLINT
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
#include "fastdeploy/vision/faceid/contrib/adaface/postprocessor.h"
#include "fastdeploy/vision/faceid/contrib/adaface/preprocessor.h"

namespace fastdeploy {
namespace vision {
namespace faceid {
/*! @brief AdaFace model object used when to load a AdaFace model exported by AdaFace.
 */
class FASTDEPLOY_DECL AdaFace : public FastDeployModel {
 public:
  /** \brief  Set path of model file and the configuration of runtime.
   *
   * \param[in] model_file Path of model file, e.g ./adaface.onnx
   * \param[in] params_file Path of parameter file, e.g adaface/model.pdiparams, if the model format is ONNX, this parameter will be ignored
   * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in "valid_cpu_backends"
   * \param[in] model_format Model format of the loaded model, default is ONNX format
   */
  AdaFace(
      const std::string& model_file, const std::string& params_file = "",
      const RuntimeOption& custom_option = RuntimeOption(),
      const ModelFormat& model_format = ModelFormat::ONNX);

  std::string ModelName() const { return "insightface_rec"; }

  /** \brief Predict the detection result for an input image
   *
   * \param[in] img The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
   * \param[in] result The output FaceRecognitionResult will be writen to this structure
   * \return true if the prediction successed, otherwise false
   */
  virtual bool Predict(const cv::Mat& im, FaceRecognitionResult* result);

  /** \brief Predict the detection results for a batch of input images
   *
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output FaceRecognitionResult list
   * \return true if the prediction successed, otherwise false
   */
  virtual bool BatchPredict(const std::vector<cv::Mat>& images,
                            std::vector<FaceRecognitionResult>* results);

  /// Get preprocessor reference of AdaFace
  virtual AdaFacePreprocessor& GetPreprocessor() {
    return preprocessor_;
  }

  /// Get postprocessor reference of AdaFace
  virtual AdaFacePostprocessor& GetPostprocessor() {
    return postprocessor_;
  }

 protected:
  bool Initialize();
  AdaFacePreprocessor preprocessor_;
  AdaFacePostprocessor postprocessor_;
};

}  // namespace faceid
}  // namespace vision
}  // namespace fastdeploy
