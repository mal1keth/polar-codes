#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <boost/random.hpp>

//----------------------------------------------------------------------------------------------------
/**
 * Reliability sequence: Precomputed permuted ordering of channels from worst to best upto N = 1024.
 * Hardcoded for efficiency. 
 */
std::vector<int> rel = {0,1,2,4,8,16,32,3,5,64,9,6,17,10,18,128,12,33,65,20,256,34,24,36,7,129,66,512,11,40,68,130,
    19,13,48,14,72,257,21,132,35,258,26,513,80,37,25,22,136,260,264,38,514,96,67,41,144,28,69,42,
    516,49,74,272,160,520,288,528,192,544,70,44,131,81,50,73,15,320,133,52,23,134,384,76,137,82,
    56,27,97,39,259,84,138,145,261,29,43,98,515,88,140,30,146,71,262,265,161,576,45,100,640,51,
    148,46,75,266,273,517,104,162,53,193,152,77,164,768,268,274,518,54,83,57,521,112,135,78,289,
    194,85,276,522,58,168,139,99,86,60,280,89,290,529,524,196,141,101,147,176,142,530,321,31,200,
    90,545,292,322,532,263,149,102,105,304,296,163,92,47,267,385,546,324,208,386,150,153,165,106,
    55,328,536,577,548,113,154,79,269,108,578,224,166,519,552,195,270,641,523,275,580,291,59,169,
    560,114,277,156,87,197,116,170,61,531,525,642,281,278,526,177,293,388,91,584,769,198,172,120,
    201,336,62,282,143,103,178,294,93,644,202,592,323,392,297,770,107,180,151,209,284,648,94,204,
    298,400,608,352,325,533,155,210,305,547,300,109,184,534,537,115,167,225,326,306,772,157,656,
    329,110,117,212,171,776,330,226,549,538,387,308,216,416,271,279,158,337,550,672,118,332,579,
    540,389,173,121,553,199,784,179,228,338,312,704,390,174,554,581,393,283,122,448,353,561,203,
    63,340,394,527,582,556,181,295,285,232,124,205,182,643,562,286,585,299,354,211,401,185,396,
    344,586,645,593,535,240,206,95,327,564,800,402,356,307,301,417,213,568,832,588,186,646,404,
    227,896,594,418,302,649,771,360,539,111,331,214,309,188,449,217,408,609,596,551,650,229,159,
    420,310,541,773,610,657,333,119,600,339,218,368,652,230,391,313,450,542,334,233,555,774,175,
    123,658,612,341,777,220,314,424,395,673,583,355,287,183,234,125,557,660,616,342,316,241,778,
    563,345,452,397,403,207,674,558,785,432,357,187,236,664,624,587,780,705,126,242,565,398,346,
    456,358,405,303,569,244,595,189,566,676,361,706,589,215,786,647,348,419,406,464,680,801,362,
    590,409,570,788,597,572,219,311,708,598,601,651,421,792,802,611,602,410,231,688,653,248,369,
    190,364,654,659,335,480,315,221,370,613,422,425,451,614,543,235,412,343,372,775,317,222,426,
    453,237,559,833,804,712,834,661,808,779,617,604,433,720,816,836,347,897,243,662,454,318,675,
    618,898,781,376,428,665,736,567,840,625,238,359,457,399,787,591,678,434,677,349,245,458,666,
    620,363,127,191,782,407,436,626,571,465,681,246,707,350,599,668,790,460,249,682,573,411,803,
    789,709,365,440,628,689,374,423,466,793,250,371,481,574,413,603,366,468,655,900,805,615,684,
    710,429,794,252,373,605,848,690,713,632,482,806,427,904,414,223,663,692,835,619,472,455,796,
    809,714,721,837,716,864,810,606,912,722,696,377,435,817,319,621,812,484,430,838,667,488,239,
    378,459,622,627,437,380,818,461,496,669,679,724,841,629,351,467,438,737,251,462,442,441,469,
    247,683,842,738,899,670,783,849,820,728,928,791,367,901,630,685,844,633,711,253,691,824,902,
    686,740,850,375,444,470,483,415,485,905,795,473,634,744,852,960,865,693,797,906,715,807,474,
    636,694,254,717,575,913,798,811,379,697,431,607,489,866,723,486,908,718,813,476,856,839,725,
    698,914,752,868,819,814,439,929,490,623,671,739,916,463,843,381,497,930,821,726,961,872,492,
    631,729,700,443,741,845,920,382,822,851,730,498,880,742,445,471,635,932,687,903,825,500,846,
    745,826,732,446,962,936,475,853,867,637,907,487,695,746,828,753,854,857,504,799,255,964,909,
    719,477,915,638,748,944,869,491,699,754,858,478,968,383,910,815,976,870,917,727,493,873,701,
    931,756,860,499,731,823,922,874,918,502,933,743,760,881,494,702,921,501,876,847,992,447,733,
    827,934,882,937,963,747,505,855,924,734,829,965,938,884,506,749,945,966,755,859,940,830,911,
    871,639,888,479,946,750,969,508,861,757,970,919,875,862,758,948,977,923,972,761,877,952,495,
    703,935,978,883,762,503,925,878,735,993,885,939,994,980,926,764,941,967,886,831,947,507,889,
    984,751,942,996,971,890,509,949,973,1000,892,950,863,759,1008,510,979,953,763,974,954,879,981,
    982,927,995,765,956,887,985,997,986,943,891,998,766,511,988,1001,951,1002,893,975,894,1009,955,
    1004,1010,957,983,958,987,1012,999,1016,767,989,1003,990,1005,959,1011,1013,895,1006,1014,1017,
    1018,991,1020,1007,1015,1019,1021,1022,1023};

//----------------------------------------------------------------------------------------------------
//                                       Polar Code Class
//----------------------------------------------------------------------------------------------------
class PolarCode {
private:
    int N;
    int k;
    double SNR_dB;
    std::vector<int> Q;
    std::vector<int> melted;
    std::unordered_map<std::string, std::string> codes;

public:
    /**
     * Instantiates the (N,k) Polar Code object.
     * Args:
     *     N -> Block length
     *     k -> Message length
     *     SNR_dB -> Signal to Noise Ratio in dB
     */
    PolarCode(int N, int k, double SNR_dB = 1.0) {
        this->N = N;
        this->k = k;
        this->SNR_dB = SNR_dB;
        
        // Initialize Q with values from rel that are less than N
        for (int i : rel) {
            if (i < N) {
                Q.push_back(i);
            }
        }
        
        melted.resize(N, 0);
        for (int i = N-k; i < N; i++) {
            melted[Q[i]] = 1;
        }
    }

    //-------------------------------- Encoding  --------------------------------

    /**
     * Assigns the best k bits of the N bit channel as message bits. Rest are frozen.
     * Args: msg -> length k vector of 0s and 1s
     * Returns: u -> length N vector of 0s and 1s
     * TC: O(N), SC: O(N)
     */
    std::vector<int> src_vector(const std::vector<int>& msg) {
        std::vector<int> u(N, 0);
        for (int i = N - k; i < N; i++) {
            u[Q[i]] = msg[i - (N - k)];
        }
        return u;
    }

    /**
     * Recursively computes the polar transform on the input vector u
     * Args: u -> length N vector of 0s and 1s, n -> length of u
     * Returns: y -> length n vector of 0s and 1s
     * TC: O(N log N), SC: O(N log N)
     */
    std::vector<int> polar_transform(const std::vector<int>& u, int n) {
        if (n == 1) {
            return u;
        }
        
        // Post-order traversal
        std::vector<int> u1, u2;
        for (int i = 0; i < n; i++) {
            if (i < n/2) u1.push_back(u[i]);
            else u2.push_back(u[i]);
        }

        std::vector<int> y1 = polar_transform(u1, n/2);
        std::vector<int> y2 = polar_transform(u2, n/2);

        std::vector<int> y(n, 0);
        for (int i = 0; i < n/2; i++) {
            y[i] = y1[i] ^ y2[i];
            y[i + n/2] = y2[i];
        }
        return y;
    }

    /**
     * Converts length k message into length N codeword
     * Args: msg -> length k vector of 0s and 1s
     * Returns: y -> length N vector of 0s and 1s
     * TC: O(N log N), SC: O(N log N)
     */
    std::vector<int> encode_msg(const std::vector<int>& msg) {
        std::vector<int> u = src_vector(msg);
        std::vector<int> y = polar_transform(u, N);
        
        // Store the message and codeword
        std::string msg_str = "";
        for (int bit : msg) msg_str.push_back((char) (bit + '0'));
        
        std::string code_str = "";
        for (int bit : y) code_str.push_back((char) (bit + '0'));
        
        codes[msg_str] = code_str;
        return y;
    }
    
    //-------------------------------- Channel Transformations  --------------------------------
    
    /**
     * Converts code symbols into BPSK Symbols. 0 -> 1, 1 -> -1
     * Args: y -> length N vector of 0s and 1s
     * Returns: symbols -> length N vector of 1s and -1s
     * TC: O(N), SC: O(N)
     */
    std::vector<double> BPSK(const std::vector<int>& y) {
        std::vector<double> symbols(y.size());
        for (size_t i = 0; i < y.size(); i++) {
            symbols[i] = (y[i] == 0) ? 1.0 : -1.0;
        }
        return symbols;
    }

    /**
     * Adds AWGN of given SNR_dB to the input symbols
     * Args: symbols -> length N vector of 1s and -1s, SNR_dB -> Signal to Noise Ratio in dB
     * Returns: pair of noisy_symbols -> length N vector of real numbers, sigma_2 -> variance of noise
     * TC: O(N), SC: O(N)
     */
    std::pair<std::vector<double>, double> AWGN(const std::vector<double>& symbols, double SNR_dB) {
        double SNR_linear = std::pow(10.0, SNR_dB/10.0);
        double sigma_2 = 1.0/SNR_linear;

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        boost::random::mt19937 generator(seed);

        boost::random::normal_distribution<double> dist(0.0, std::sqrt(sigma_2));
        
        std::vector<double> noisy_symbols(symbols.size());
        for (size_t i = 0; i < symbols.size(); i++) {
            noisy_symbols[i] = symbols[i] + dist(generator);
        }
        
        return {noisy_symbols, sigma_2};
    }

    /**
     * Converts Noisy BPSK symbols into LLR values
     * Args: y -> length N vector of real numbers, sigma_2 -> variance of noise
     * Returns: LLRs -> length N vector of real numbers
     * TC: O(N), SC: O(N)
     */
    std::vector<double> LLR(const std::vector<double>& y, double sigma_2) {
        std::vector<double> r(y.size());
        for (size_t i = 0; i < y.size(); i++) {
            r[i] = 2.0 * y[i] / sigma_2;
        }
        return r;
    }

    //-------------------------------- Decoding Helpers --------------------------------
    
    /**
     * Returns the sign of the input value
     * Args: x -> real number 
     * Returns: b \in {-1, 1}
     * TC: O(1), SC: O(1)
     */
    int sign(double x) {
        return (x >= 0) ? 1 : -1;
    }

    /**
     * Assigns the product of the signs to the lowest absolute value of the input values
     * Args: r1 -> length N vector of real numbers, r2 -> length N vector of real numbers
     * Returns: product -> length N vector of real numbers
     * TC: O(N), SC: O(N)
     */
    std::vector<double> f(const std::vector<double>& r1, const std::vector<double>& r2) {
        if (r1.size() != r2.size()) {
            throw std::invalid_argument("f: r1 and r2 must have the same length");
        }
        
        std::vector<double> result(r1.size());
        for (size_t i = 0; i < r1.size(); i++) {
            result[i] = sign(r1[i]) * sign(r2[i]) * std::min(std::abs(r1[i]), std::abs(r2[i]));
        }
        return result;
    }

    /**
     * Computes r1 + r2 if b = 0, else r1 - r2
     * Args: r1 -> length N vector of real numbers, r2 -> length N vector of real numbers, b -> length N vector of 0s and 1s
     * Returns: product -> length N vector of real numbers
     * TC: O(N), SC: O(N)
     */
    std::vector<double> g(const std::vector<double>& r1, const std::vector<double>& r2, const std::vector<int>& b) {
        if (r1.size() != r2.size()) {
            throw std::invalid_argument("g: r1 and r2 must have the same length");
        }
        if (b.size() != r1.size()) {
            throw std::invalid_argument("g: b and r1 must have the same length");
        }
        
        std::vector<double> result(r1.size());
        for (size_t i = 0; i < r1.size(); i++) {
            result[i] = r2[i] + (1 - 2 * b[i]) * r1[i];
        }
        return result;
    }

    //-------------------------------- Successive Cancellation Decoding --------------------------------
    
    /**
     * Recursively transforms LLR values of (possible) noisy symbols to a polar codeword
     * Args: n -> length of L, L -> length N vector of real numbers, melted -> length N vector of 0s and 1s
     * Returns: u_hat -> length N vector of 0s and 1s
     * TC: O(N log N), SC: O(N log N)
     */
    std::vector<int> SCD(int n, const std::vector<double>& L, const std::vector<int>& melted) {
        if (n == 1) {
            if (melted[0] == 1) {
                // Hard decision
                return {(L[0] < 0) ? 1 : 0};
            } else {
                // Frozen bit, no information
                return {0};
            }
        }
        
        std::vector<double> a, b;
        std::vector<int> melted_l, melted_r;
        for (int i = 0; i < n; i++) {
            if (i < n/2) {
                a.push_back(L[i]);
                melted_l.push_back(melted[i]);
            } else {
                b.push_back(L[i]);
                melted_r.push_back(melted[i]);
            }
        }

        // ---- Recurse ----
        std::vector<int> u_left = SCD(n/2, f(a, b), melted_l);
        std::vector<int> u_right = SCD(n/2, g(a, b, u_left), melted_r);

        // ---- Combine ----
        std::vector<int> u_hat(n);
        for (int i = 0; i < n/2; i++) {
            u_hat[i] = u_left[i] ^ u_right[i];
            u_hat[i + n/2] = u_right[i];
        }
        return u_hat;
    }
    
    /**
     * Recursively transforms a polar codeword into a vector where the k best bits are the message bits
     * Args: n -> length of u, u -> length N vector of 0s and 1s, melted -> length N vector of 0s and 1s
     * Returns: u_hat -> length N vector of 0s and 1s
     * TC: O(N log N), SC: O(N log N)
     */
    std::vector<int> decode_codeword(const std::vector<int>& u, int n, const std::vector<int>& melted) {
        if (n == 1) {
            return u;
        }
        
        std::vector<int> u_first_half, u_second_half;
        std::vector<int> melted_l, melted_r;
        for (int i = 0; i < n; i++) {
            if (i < n/2) {
                u_first_half.push_back(u[i]);
                melted_l.push_back(melted[i]);
            } else {
                u_second_half.push_back(u[i]);
                melted_r.push_back(melted[i]);
            }
        }

        // ---- Recurse ----
        std::vector<int> u_xor = decode_codeword(u_first_half, n/2, melted_l);
        std::vector<int> u_right = decode_codeword(u_second_half, n/2, melted_r);

        // ---- Combine ----
        std::vector<int> u_left(n/2);
        for (int i = 0; i < n/2; i++) {
            u_left[i] = u_xor[i] ^ u_right[i];
        }
        
        std::vector<int> result(n);
        std::copy(u_left.begin(), u_left.end(), result.begin());
        std::copy(u_right.begin(), u_right.end(), result.begin() + n/2);
        
        return result;
    }

    /**
     * Applies the channel transformation to the codeword to get noisy symbols. 
     * Computes LLRs of the noisy symbols.
     * Applies SCD to the LLRs to get a polar codeword.
     * Decodes the polar codeword into a message
     * Args: y -> length N vector of 0s and 1s, SNR_dB -> (optional) Signal to Noise Ratio in dB
     * Returns: pair of d_msg -> length k string of 0s and 1s, correctness -> boolean
     * TC: O(N log N), SC: O(N log N)
     */
    std::pair<std::string, bool> decode_msg(const std::vector<int>& y, double SNR_dB = -1.0) {
        if (SNR_dB != -1.0) {
            this->SNR_dB = SNR_dB;
        }
        
        //---- Channel Transformations ----
        std::vector<double> symbols = BPSK(y);
        auto [noisy_symbols, sigma_2] = AWGN(symbols, this->SNR_dB);
        std::vector<double> llr = LLR(noisy_symbols, sigma_2);

        //---- SCD ----
        std::vector<int> y_prime = SCD(N, llr, melted);

        //---- Codeword Decoding ----
        std::vector<int> decoded_u = decode_codeword(y_prime, N, melted);
        
        //---- Message Reconstruction ----
        std::vector<int> d_msg_vec(k);
        for (int i = N - k; i < N; i++) {
            d_msg_vec[i - (N - k)] = decoded_u[Q[i]];
        }
        
        std::string d_msg = "";
        for (int bit : d_msg_vec)  d_msg.push_back((char) (bit + '0'));
        
        std::string y_str = "";
        for (int bit : y) y_str.push_back((char) (bit + '0'));
        
        bool correctness = true;
        if (codes.find(d_msg) == codes.end() || codes[d_msg] != y_str) {
            correctness = false;
        }
        
        return {d_msg, correctness};
    }
};
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
