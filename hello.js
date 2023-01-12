const path = require('path');
const { vad } = require(path.join(__dirname, 'bin/Darwin/arm64/vad_addon'));

console.log(vad(path.join(__dirname, 'silero_vad.onnx'), path.join(__dirname, 'samples.wav')));
