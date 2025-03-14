#include <iostream>
#include <vector>
#include "polar_codes.cpp"
#include <unordered_map>
#include <boost/random.hpp>
#include <chrono>

const int N = 1024;
const double eps = 1e-6;

std::vector<int> random_msg(int k){

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    boost::random::mt19937 generator(seed);
    boost::random::bernoulli_distribution<> dist(0.5);

    std::vector<int> msg(k);
    for (int i = 0; i < k; i++) {
        msg[i] = dist(generator);
    }
    return msg;
}

std::unordered_map<int, int> success_rate(int k, int samples = 1000) {
    PolarCode pc(N, k);
    std::unordered_map<int, int> success_rate;
    for(double SNR_dB = 0.1; SNR_dB <= 15; SNR_dB += 0.1){
        for (int i = 0; i < samples; i++) {
            std::vector<int> msg = random_msg(k);
            auto y = pc.encode_msg(msg);
            auto [d_msg, correct] = pc.decode_msg(y, SNR_dB);
            if (correct) {
                success_rate[int(10 * SNR_dB)]++;
            }
        }
        success_rate[int(10 * SNR_dB)] /= samples;
    }
    return success_rate;
}

int main(){
    int k = 2;
    auto start = std::chrono::high_resolution_clock::now();
    for(int k = 1; k <= 256; k *= 2){
        auto s_r = success_rate(k);
        std::cout << "k = " << k << std::endl;
        for(auto &[snr, success] : s_r){
            std::cout << ((double)snr / 10) << ": " << success << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds" << std::endl;
}