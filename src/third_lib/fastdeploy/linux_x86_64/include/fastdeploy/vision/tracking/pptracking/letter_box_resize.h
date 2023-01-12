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

#include "fastdeploy/vision/common/processors/base.h"

namespace fastdeploy {
namespace vision {

class LetterBoxResize : public Processor {
 public:
  LetterBoxResize(const std::vector<int>& target_size,
                  const std::vector<float>& color) {
    target_size_ = target_size;
    color_ = color;
  }

  std::string Name() { return "LetterBoxResize"; }

  virtual bool operator()(Mat* mat, ProcLib lib = ProcLib::DEFAULT);

  static bool Run(Mat* mat, const std::vector<int>& target_size,
                const std::vector<float>& color,
                ProcLib lib = ProcLib::DEFAULT);

 private:
  std::vector<int> target_size_;
  std::vector<float> color_;
};
}  // namespace vision
}  // namespace fastdeploy
