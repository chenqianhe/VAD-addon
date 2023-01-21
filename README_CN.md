[English](README.md) | 简体中文

# VAD-addon

该项目提供了一个可以**在 node 和 electron 环境中执行VAD模型推理的插件**，基于 [cmake-js](https://github.com/cmake-js/cmake-js) 和 [Fastdeploy](https://github.com/PaddlePaddle/FastDeploy)。
[**Silero VAD**](https://github.com/snakers4/silero-vad) 是一个预训练的企业级语音活动探测器。

我们的项目支持在 Windows AMD64(x86_64)、 macOS x86_64、 macOS arm64、 Linux AMD64(x86_64)、 Linux aarch64(arm64) 中运行。

## 安装

项目使用了 [**cmake**](https://cmake.org/) 和 [**npm**](https://www.npmjs.com/)(在 Windows 中，我们额外需要 [**Visual Studio**](https://visualstudio.microsoft.com))。请检查它们是否已被安装。

```bash
git clone https://github.com/chenqianhe/VAD-addon
cd VAD-addon
npm install
```

### Windows 系统

1. 你可以参考 [Visual Studio 2019 创建 CMake 工程使用 C++ SDK](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/use_sdk_on_windows.md#VisualStudio2019)进行配置。
2. 你也可以使用 [Clion](https://www.jetbrains.com/clion/) 进行配置。
<details>
<summary>Clion 配置</summary>
<img width="1425" alt="image" src="https://user-images.githubusercontent.com/54462604/213860521-5cf830ef-fa95-460f-8b0a-e44f95a56070.png">
<img alt="image" src="https://user-images.githubusercontent.com/54462604/213860779-46da3900-88f2-408b-950b-5e920c4b744b.png">

</details>

## 用法

### Default for node addon

```bash
npx gulp
```

### Electron addon

1. 根据 [cmake-js 相关配置](https://github.com/cmake-js/cmake-js#configuration) 进行配置。
2. 运行
```bash
npx gulp
```

### 使用 cmake-js options

cmake-js 提供一些 [options](https://github.com/cmake-js/cmake-js#installation)，你可以像使用 cmake-js 一样直接使用这些 options 。

```bash
npx gulp [options]
```
> 示例(当使用 Clion 时，我们通常需要指定特殊 cmake 路径)
> ```bash
> npx gulp -c xxx/cmake
> ```

### 运行 node 环境下示例
```bash
node index.js
```
> 结果类似
> ```
>[
>  { start: '0.000000', end: '2.304000' },
>  { start: '5.056000', end: '7.680000' },
>  { start: '8.320001', end: '10.496000' }
>]
>```

## 参考

- [FastDeploy](https://github.com/PaddlePaddle/FastDeploy/)
- [cmake-js](https://github.com/cmake-js/cmake-js/)
- [silero-vad](https://github.com/snakers4/silero-vad)

## LICENSE

[MIT](LICENSE)
