English | [简体中文](README_CN.md)

# VAD-addon

This repo provides an addon that can **perform VAD model reasoning in nodes and electric environments**, based on [cmake-js](https://github.com/cmake-js/cmake-js) and [Fastdeploy](https://github.com/PaddlePaddle/FastDeploy).
[**Silero VAD**](https://github.com/snakers4/silero-vad) is a pre-trained enterprise-grade Voice Activity Detector. 

Our project supports Windows AMD64(x86_64), macOS x86_64, macOS arm64, Linux AMD64(x86_64), Linux aarch64(arm64)

## Install

This project uses [**cmake**](https://cmake.org/) and [**npm**](https://www.npmjs.com/).(In Windows, we need [**Visual Studio**](https://visualstudio.microsoft.com) additionally) Go check them out if you don't have them locally installed.

```bash
git clone https://github.com/chenqianhe/VAD-addon

cd VAD-addon

npm install
```

### In Windows

1. You can refer to [Visual Studio 2019 Create CMake project using C++ SDK](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/en/faq/use_sdk_on_windows.md#33-visual-studio-2019-create-cmake-project-using-c-sdk) for configuration.
2. You can also use Clion to configure.
<details>
<summary>Clion configuration</summary>
<img width="1425" alt="image" src="https://user-images.githubusercontent.com/54462604/213860521-5cf830ef-fa95-460f-8b0a-e44f95a56070.png">
<img alt="image" src="https://user-images.githubusercontent.com/54462604/213860779-46da3900-88f2-408b-950b-5e920c4b744b.png">

</details>

## Usgae

