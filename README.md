English | [简体中文](README_CN.md)

# VAD-addon

This project provides an addon that can **perform VAD model reasoning in node and electron environments**, based on [cmake-js](https://github.com/cmake-js/cmake-js) and [Fastdeploy](https://github.com/PaddlePaddle/FastDeploy).
[**Silero VAD**](https://github.com/snakers4/silero-vad) is a pre-trained enterprise-grade Voice Activity Detector. 

Our project supports Windows AMD64(x86_64), macOS x86_64, macOS arm64, Linux AMD64(x86_64), Linux aarch64(arm64).

## Install

This project uses [**cmake**](https://cmake.org/) and [**npm**](https://www.npmjs.com/)(In Windows, we need [**Visual Studio**](https://visualstudio.microsoft.com) additionally). Go check them out if you don't have them locally installed.

```bash
git clone https://github.com/chenqianhe/VAD-addon

cd VAD-addon

npm install
```

### In Windows

1. You can refer to [Visual Studio 2019 Create CMake project using C++ SDK](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/en/faq/use_sdk_on_windows.md#33-visual-studio-2019-create-cmake-project-using-c-sdk) for configuration.
2. You can also use [Clion](https://www.jetbrains.com/clion/) to configure.
<details>
<summary>Clion configuration</summary>
<img width="1425" alt="image" src="https://user-images.githubusercontent.com/54462604/213860521-5cf830ef-fa95-460f-8b0a-e44f95a56070.png">
<img alt="image" src="https://user-images.githubusercontent.com/54462604/213860779-46da3900-88f2-408b-950b-5e920c4b744b.png">

</details>

## Usgae

### Default for node addon

```bash
npx gulp
```

### Electron addon

1. Set the configuration according to the [relevant configuration of cmake-js](https://github.com/cmake-js/cmake-js#configuration)
2. Run
```bash
npx gulp
```

### Use cmake-js options

The cmake-js provides some [options](https://github.com/cmake-js/cmake-js#installation), and you can add it directly, just like using cmake-js.

```bash
npx gulp [options]
```
> Example(When using Clion, we usually need to appoint special cmake path)
> ```bash
> npx gulp -c xxx/cmake
> ```

### Run example in node env
```bash
node index.js
```
> Result likes
> ```
>[
>  { start: '0.000000', end: '2.304000' },
>  { start: '5.056000', end: '7.680000' },
>  { start: '8.320001', end: '10.496000' }
>]
>```


## Reference

- [FastDeploy](https://github.com/PaddlePaddle/FastDeploy/)
- [cmake-js](https://github.com/cmake-js/cmake-js/)
- [silero-vad](https://github.com/snakers4/silero-vad)

## LICENSE

[MIT](LICENSE)

