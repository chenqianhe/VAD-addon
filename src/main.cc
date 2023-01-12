#include "napi.h"
#include "vad.h"


Napi::Object vad(const Napi::CallbackInfo& info) {
    // need model path and wav path
    Napi::Env env = info.Env();
    if (info.Length() < 2) {
        Napi::Error::New(
                env, "Arg number wrong. We need 2 args; the first is model path and the second is wav path.")
        .ThrowAsJavaScriptException();
        return Napi::Object::New(env);
    }
    std::string modelPath = info[0].As<Napi::String>();
    std::string wavPath = info[1].As<Napi::String>();
    // run model
    Vad vad(modelPath);
    vad.init();
    vad.loadAudio(wavPath);
    vad.Predict();
    std::vector<std::map<std::string, float>> result = vad.getResult();

    Napi::Object res = Napi::Array::New(env, result.size());
    for (int i = 0; i < result.size(); ++i) {
        Napi::Object temp = Napi::Object::New(env);
        temp["start"] = Napi::String::New(env, std::to_string(result[i]["start"]));
        temp["end"] = Napi::String::New(env, std::to_string(result[i]["end"]));
        res[i] = temp;
    }
    Napi::String num = Napi::String::New(env, modelPath+wavPath);
    return res;
}


Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set(
            Napi::String::New(env, "vad"),
            Napi::Function::New(env, vad)
    );
    return exports;
}

NODE_API_MODULE(vad, Init);
