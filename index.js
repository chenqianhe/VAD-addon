const path = require('path');
// mac arm
// const { vad } = require(path.join(__dirname, 'bin/Darwin/arm64/vad_addon'));
// win amd64
// const { vad } = require(path.join(__dirname, 'bin/Windows/AMD64/vad_addon'));
// linux aarch64
const { vad } = require(path.join(__dirname, 'bin/Linux/aarch64/vad_addon'));

console.log(vad(path.join(__dirname, 'silero_vad.onnx'), path.join(__dirname, 'samples.wav')));
