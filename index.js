const path = require('path');
const { vad } = require(path.join(__dirname, 'build/Release/vad_addon'));

console.log(vad(path.join(__dirname, 'silero_vad.onnx'), path.join(__dirname, 'samples.wav')));
