// Standard llama C API client: complete raw vocabulary, no sampler/top-k/engine patch.
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Answer { std::string label; int id; };
static void require(bool condition) { if (!condition) throw std::runtime_error("invalid_input_or_logits"); }
static double logsumexp(const std::vector<double> &values) {
    require(!values.empty());
    const double maximum = *std::max_element(values.begin(), values.end());
    double sum = 0;
    for (double x : values) { require(std::isfinite(x)); sum += std::exp(x - maximum); }
    return maximum + std::log(sum);
}
static void emit(const float *raw, int vocab, const std::vector<Answer> &answers) {
    require(raw && vocab > 0 && answers.size() >= 3);
    std::vector<double> full(raw, raw + vocab), joint, options;
    std::set<int> ids; std::set<std::string> labels;
    int u = 0;
    for (const auto &a : answers) {
        require(a.id >= 0 && a.id < vocab && ids.insert(a.id).second && labels.insert(a.label).second);
        require(a.label == "U" || a.label == "yes" || a.label == "no" ||
                (a.label.size() == 1 && ((a.label[0] >= 'A' && a.label[0] <= 'T') ||
                 (a.label[0] >= '0' && a.label[0] <= '9'))));
        joint.push_back(raw[a.id]);
        if (a.label == "U") ++u; else options.push_back(raw[a.id]);
    }
    require(u == 1);
    const double zfull=logsumexp(full), zjoint=logsumexp(joint), zoptions=logsumexp(options);
    std::cout << std::setprecision(17) << "{\"logits_kind\":\"raw_logits\",\"vocabulary_size\":" << vocab
              << ",\"sampler_used\":false,\"calibrated\":false,\"answers\":{";
    for (size_t i=0; i<answers.size(); ++i) {
        const auto &a=answers[i]; const double x=raw[a.id];
        if (i) std::cout << ',';
        std::cout << '"' << a.label << "\":{\"token_id\":" << a.id << ",\"raw_logit\":" << x
                  << ",\"full_vocabulary_probability\":" << std::exp(x-zfull)
                  << ",\"joint_legal_plus_U_probability\":" << std::exp(x-zjoint);
        if (a.label != "U") std::cout << ",\"conditional_probability_without_U\":" << std::exp(x-zoptions);
        std::cout << '}';
    }
    std::cout << "}}\n";
}
static std::vector<std::string> split(const std::string &input) {
    std::stringstream s(input); std::string value; std::vector<std::string> result;
    while (std::getline(s,value,',')) { require(!value.empty()); result.push_back(value); }
    return result;
}
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            const float raw[]={float(std::log(.1)),float(std::log(.2)),float(std::log(.2)),float(std::log(.5))};
            bool rejected=false;
            try { emit(raw,4,{{"A",0},{"B",0},{"U",2}}); } catch (...) { rejected=true; }
            require(rejected);
            float invalid[]={raw[0],raw[1],raw[2],std::numeric_limits<float>::quiet_NaN()};
            rejected=false;
            try { emit(invalid,4,{{"A",0},{"B",1},{"U",2}}); } catch (...) { rejected=true; }
            require(rejected);
            emit(raw,4,{{"A",0},{"B",1},{"U",2}}); return 0;
        }
        require(argc == 7 && std::string(argv[1]) == "--model" && std::string(argv[3]) == "--tokens"
                && std::string(argv[5]) == "--answers");
        std::vector<llama_token> tokens;
        for (const auto &s:split(argv[4])) { size_t n=0; int x=std::stoi(s,&n); require(n==s.size() && x>=0); tokens.push_back(x); }
        require(!tokens.empty() && tokens.size() <= 4095);
        std::vector<Answer> answers;
        for (const auto &s:split(argv[6])) { auto pos=s.find(':'); require(pos!=std::string::npos); size_t n=0;
            int id=std::stoi(s.substr(pos+1),&n); require(n==s.size()-pos-1); answers.push_back({s.substr(0,pos),id}); }
        llama_backend_init();
        auto mp=llama_model_default_params(); mp.n_gpu_layers=0;
        auto *model=llama_model_load_from_file(argv[2],mp); require(model);
        const auto *vocab=llama_model_get_vocab(model);
        for (auto token:tokens) require(token<llama_vocab_n_tokens(vocab));
        auto cp=llama_context_default_params(); cp.n_ctx=std::max<size_t>(512,tokens.size()+1); cp.n_batch=cp.n_ctx;
        cp.n_threads=4; cp.n_threads_batch=4;
        auto *ctx=llama_init_from_model(model,cp); require(ctx);
        auto batch=llama_batch_get_one(tokens.data(),int32_t(tokens.size()));
        require(llama_decode(ctx,batch)==0);
        // This getter synchronizes and returns the complete unsampled n_vocab row.
        emit(llama_get_logits_ith(ctx,-1),llama_vocab_n_tokens(vocab),answers);
        llama_free(ctx); llama_model_free(model); llama_backend_free(); return 0;
    } catch (...) { std::cerr << "raw_logits_probe_failed\n"; return 2; }
}
