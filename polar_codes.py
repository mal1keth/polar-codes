from numpy import random, sqrt

#----------------------------------------------------------------------------------------------------
'''
Reliability sequence: Precomputed permuted ordering of channels from worst to best upto N = 1024.
Hardcoded for efficiency. 
'''
rel = [0,1,2,4,8,16,32,3,5,64,9,6,17,10,18,128,12,33,65,20,256,34,24,36,7,129,66,512,11,40,68,130,
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
	1018,991,1020,1007,1015,1019,1021,1022,1023]

#----------------------------------------------------------------------------------------------------
#                                       Polar Code Class
#----------------------------------------------------------------------------------------------------
class PolarCode:
    '''
    Instantiates the (N,k) Polar Code object.
    Args :
        N -> Block length
        k -> Message length
        SNR_dB -> Signal to Noise Ratio in dB
    '''
    def __init__(self, N, k, SNR_dB = 1):
        self.N = N
        self.k = k
        self.SNR_dB = SNR_dB
        

        self.Q = [i for i in rel if i < N]
        self.melted= [0] * N

        for i in range(N-k, N):
            self.melted[self.Q[i]] = 1

        self.codes = {}
    #-------------------------------- Encoding  --------------------------------

    #--------------------------
    '''
    Assigns the best k bits of the N bit channel as message bits. Rest are frozen.
    Args : msg -> length k list of 0s and 1s
    Returns : u -> length N list of 0s and 1s
    TC : O(N), SC : O(N)
    '''
    def src_vector(self, msg):
        u = [0] * self.N
        for i in range(self.N - self.k, self.N):
            u[self.Q[i]] = msg[i - (self.N - self.k)]
        return u
    #---------------- Polar Transform ---------------
    '''
    Recursively computes the polar transform on the input vector u
    Args : u -> length N list of 0s and 1s, n -> len(m)
    Returns : y -> length n list of 0s and 1s
    TC : O(N log N), SC : O(N log N)
    '''
    def polar_transform(self, u, n):
        if(n == 1):
            return u
        
        # Post-order traversal
        y_1 = self.polar_transform(u[:n//2], n//2)
        y_2 = self.polar_transform(u[n//2:], n//2)

        y = [0] * n
        for i in range(n//2):
            y[i] = y_1[i] ^ y_2[i]
        for i in range(n//2):
            y[i + n//2] = y_2[i]
        return y
    #--------------------------
    '''
    Converts length k message into length N codeword
    Args : msg -> length k list of 0s and 1s
    Returns : y -> length N list of 0s and 1s
    TC : O(N log N), SC : O(N log N)
    '''
    def encode_msg(self, msg):
        u = self.src_vector(msg)
        y = self.polar_transform(u, self.N)
        self.codes[''.join(str(x) for x in msg)] = ''.join(str(x) for x in y)
        return y
    
    #-------------------------------- Channel Transformations  --------------------------------
    '''
    Converts code symbols into BPSK Symbols. 0 -> 1, 1 -> -1
    Args : y -> length N list of 0s and 1s
    Returns : symbols -> length N list of 1s and -1s
    TC : O(N), SC : O(N)
    '''
    def BPSK(self, y):
        return [1 if y[i] == 0 else -1 for i in range(len(y))]
    #--------------------------
    '''
    Adds AWGN of given SNR_dB to the input symbols
    Args : symbols -> length N list of 1s and -1s, SNR_dB -> Signal to Noise Ratio in dB
    Returns : noisy_symbols -> length N list of real numbers, sigma_2 -> variance of noise
    TC : O(N), SC : O(N)
    '''
    def AWGN(self, symbols, SNR_dB):
        SNR_linear = 10**(SNR_dB/10)
        sigma_2 = 1/SNR_linear

        noise = random.normal(0, sqrt(sigma_2), len(symbols))
        noisy_symbols = [symbols[i] + noise[i] for i in range(len(symbols))]

        return noisy_symbols, sigma_2
    #--------------------------
    '''
    Converts Noisy BPSK symbols into LLR values
    Args : noisy_symbols -> length N list of real numbers, sigma_2 -> variance of noise
    Returns : LLRs -> length N list of real numbers
    TC : O(N), SC : O(N)
    '''
    def LLR(self, y, sigma_2):
        r = [2*symbol/sigma_2 for symbol in y]
        return r
    #-------------------------------- Decoding Helpers --------------------------------
    '''
    Returns the sign of the input value
    Args : x -> real number 
    Returns : b \in {-1, 1}
    TC : O(1), SC : O(1)
    '''
    def sign(self, x):
        return 1*(x >= 0) - 1*(x < 0)
    #--------------------------
    '''
    Assigns the product of the signs to the lowest absolute value of the input values
    Args : r1 -> length N list of real numbers, r2 -> length N list of real numbers
    Returns : product -> length N list of real numbers
    TC : O(N), SC : O(N)
    '''
    def f(self, r1, r2):
        if(len(r1) != len(r2)):
            raise ValueError("f: r1 and r2 must have the same length")
        return [(self.sign(r1[i]) * self.sign(r2[i]) * min(abs(r1[i]), abs(r2[i]))) for i in range(len(r1))]
    #--------------------------
    '''
    Computes r1 + r2 if b = 0, else r1 - r2
    Args : r1 -> length N list of real numbers, r2 -> length N list of real numbers, b -> length N list of 0s and 1s
    Returns : product -> length N list of real numbers
    TC : O(N), SC : O(N)
    '''
    def g(self, r1, r2, b):
        if(len(r1) != len(r2)):
            raise ValueError("g: r1 and r2 must have the same length")
        if(len(b) != len(r1)):
            raise ValueError("g: b and r1 must have the same length")
        return [( (r2[i] + (1 - 2 * b[i]) * r1[i]) )  for i in range(len(r1))]

    #-------------------------------- Successive Cancellation Decoding --------------------------------
    '''
    Recursively transforms LLR values of (possible) noisy symbols to a polar codeword
    Args : n -> len(L) == N, L -> length N list of real numbers, melted -> length N list of 0s and 1s
    Returns : u_hat -> length N list of 0s and 1s
    TC : O(N log N), SC : O(N log N)
    '''
    def SCD(self, n, L, melted):
        if(n == 1):
            if(melted[0] == 1):
                # Hard decision
                return [1] if L[0] < 0 else [0]
            else:
                # Frozen bit, no information
                return [0]
        a = L[:n//2]
        b = L[n//2: n]

        # ---- Recurse ----
        u_left = self.SCD(n//2, self.f(a, b), melted[:n//2])
        u_right = self.SCD(n//2, self.g(a, b, u_left), melted[n//2:])

        # ---- Combine ----
        u_hat = [u_left[i] ^ u_right[i] for i in range(n//2)] + u_right
        return u_hat
    
    #--------------------------
    '''
    Recursively transforms a polar codeword into a vector where the k best bits are the message bits
    Args : n -> len(u) == N, u -> length N list of 0s and 1s, melted -> length N list of 0s and 1s
    Returns : u_hat -> length N list of 0s and 1s
    TC : O(N log N), SC : O(N log N)
    '''
    def decode_codeword(self, u, n, melted):
        if(n == 1):
            return u
        # ---- Recurse ----
        u_xor = self.decode_codeword(u[:n//2], n//2, melted[:n//2])
        u_right = self.decode_codeword(u[n//2:], n//2, melted[n//2:])

        # ---- Combine ----
        u_left = [a ^ b for a, b in zip(u_xor, u_right)]
        return u_left + u_right
    #--------------------------
    '''
    Applies the channel transformation to the codeword to get noisy symbols. 
    Computes LLRs of the noisy symbols.
    Applies SCD to the LLRs to get a polar codeword.
    Decodes the polar codeword into a message
    Args : y -> length N list of 0s and 1s, SNR_dB -> (optional) Signal to Noise Ratio in dB
    Returns : d_msg -> length k list of 0s and 1s, correctness -> boolean
    TC : O(N log N), SC : O(N log N)
    '''
    def decode_msg(self, y, SNR_dB = -1):

        if(SNR_dB != -1):
            self.SNR_dB = SNR_dB
        
        #---- Channel Transformations ----
        sym, sigma_2 = self.AWGN(self.BPSK(y), self.SNR_dB)
        llr = self.LLR(sym, sigma_2)

        #---- SCD ----
        y_prime = self.SCD(self.N, llr, self.melted)

        #---- Codeword Decoding ----
        decoded_u = self.decode_codeword(y_prime, self.N, self.melted)
        
        #---- Message Reconstruction ----
        d_msg = [0] * self.k
        for i in range(self.N - self.k, self.N):
            d_msg[i - (self.N - self.k)] = decoded_u[self.Q[i]]
        d_msg = ''.join(str(x) for x in d_msg)

        correctness = True
        if(d_msg not in self.codes or self.codes[d_msg] != ''.join(str(x) for x in y)):
            correctness = False
        
        return d_msg, correctness
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------