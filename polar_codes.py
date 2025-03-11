import random
import math
import numpy as np
import sys


#--------------------------- Reliability Sequence ---------------------------

with open('reliability.txt', 'r') as file:
    sequence = [int(x) for x in file.read().split(',')]

#-------------------------------- Input Parameters --------------------------------

# TO-DO : Take terminal inputs N = 2^n, k <= N
# TO-DO : Put this all in a class object called PolarCode. Thus, to each N,k
# we can associate a PolarCode object, precisely the (N,k) polar code.
# TO-DO : Add rate calculation 
# TO-DO : Graph rate of success as a function of SNR_dB
# TO-DO : Improve comments / documentations
# TO-DO : Improve Testing
N = 128
k = 64
SNR_dB = 5


#--------------------------------- msg -> u ---------------------------------
def src_vector(msg, N, k, Q):
    u = [0] * N
    for i in range(N - k, N):
        u[Q[i]] = msg[i - (N - k)]
    return u
#-------------------------------- Correctness Checks --------------------------------
# O(N^2)
def encoding_check(u, N):

    n = int(math.log(N, 2))
    G = np.array([[1, 0], [1, 1]])
    G_n = G

    for i in range(1, n):
        G_n = np.kron(G, G_n)

    u_np = np.array(u)
    return np.mod(u_np.dot(G_n), 2).tolist()

#-------------------------------- Encoding --------------------------------

# O(N log N)
# Recursively encodes left and right halves of u to get u_1, u_2
# Returns [u_1 ^ u_2, u_2]
def encode(m, N, k, Q):
    if(N == 1):
        return m
    #---- Head Recursion ----
    # Depth-first / Post-order traversal
    u_1 = encode(m[:N//2], N//2, k, Q)
    u_2 = encode(m[N//2:], N//2, k, Q)

    u = [0] * N
    for i in range(N//2):
        u[i] = u_1[i] ^ u_2[i]
    for i in range(N//2):
        u[i + N//2] = u_2[i]
    return u


#-------------------------------- Decoding --------------------------------

#----- Decoding Helpers -----
def sign(x):
    return 1*(x >= 0) - 1*(x < 0)

def f(r1, r2):
    if(len(r1) != len(r2)):
        raise ValueError("f: r1 and r2 must have the same length")
    return [(sign(r1[i]) * sign(r2[i]) * min(abs(r1[i]), abs(r2[i]))) for i in range(len(r1))]

def g(r1, r2, b):
    if(len(r1) != len(r2)):
        raise ValueError("g: r1 and r2 must have the same length")
    if(len(b) != len(r1)):
        raise ValueError("g: b and r1 must have the same length")
    return [( (r2[i] + (1 - 2 * b[i]) * r1[i]) )  for i in range(len(r1))]

#----------------------------


# Don't need to maintain explicit L matrix 
# Or node updates / ucap arrays since we do it recursively
# O(N log N)
def SCD(N, L, melted):
    if(N == 1):
        if(melted[0] == 1):
            # Hard decision
            return [1] if L[0] < 0 else [0]
        else:
            # Frozen bit, no information
            return [0]
    a = L[:N//2]
    b = L[N//2: N]

    #---- Head Recursion ----
    # Post-order traversal
    
    u_left = SCD(N//2, f(a, b), melted[:N//2])
    u_right = SCD(N//2, g(a, b, u_left), melted[N//2:])

    u_hat = [0] * N
    for i in range(N//2):
        u_hat[i] = (u_left[i] + u_right[i]) % 2
    for i in range(N//2):
        u_hat[i + N//2] = u_right[i]
    return u_hat

def decode_codeword(u, N, melted):
    if(N == 1):
        return u
    # Post-order traversal
    u_xor = decode_codeword(u[:N//2], N//2, melted[:N//2])
    u_right = decode_codeword(u[N//2:], N//2, melted[N//2:])
    u_left = [a ^ b for a, b in zip(u_xor, u_right)]
    return u_left + u_right

#-------------------------------- Channel Modelling --------------------------------
# Sends 0 to 1 and 1 to -1
def BPSK(y):
    return [1 if y[i] == 0 else -1 for i in range(len(y))]

# Scavenged from existing github implementations
# Adds AWGN noise to BPSK symbols
def AWGN(symbols, SNR_dB):
    
    # Convert SNR from dB to linear scale
    SNR_linear = 10**(SNR_dB/10)
    
    # Note: 
    # sigma_2 = noise variance (sigma^2)
    # For BPSK, signal power is 1, so sigma^2 = 1/SNR
    sigma_2 = 1/SNR_linear
    
    noise = np.random.normal(0, np.sqrt(sigma_2), len(symbols))
    noisy_symbols = [symbols[i] + noise[i] for i in range(len(symbols))]
    
    return noisy_symbols, sigma_2


# Computes log-likelihood ratios from (noisy) BPSK symbols
# sigma_2: noise variance
def LLR(y, sigma_2):
    r = [2*symbol/sigma_2 for symbol in y]
    return r


#-------------------------------- Testing --------------------------------

def main():
    # ---- Source Message ----
    msg = [random.randint(0, 1) for _ in range(k)]
    # Part of reliability sequence we use
    Q = [i for i in sequence if i < N]
    F = Q[:N-k]
    
    # ---- Encoding --- 
    u = src_vector(msg, N, k, Q)
    y = encode(u, N, k, Q)
    y_brute = encoding_check(u, N)
    

    # ---- Channel Transmission ----
    bpsk_symbols = BPSK(y) 
    noisy_symbols, sigma_2 = AWGN(bpsk_symbols, SNR_dB)
    llr_values = LLR(noisy_symbols, sigma_2)
    melted = src_vector([1] * k, N, k, Q)

    # ---- Decoding ----

    # This gives us back the original codeword (if successful)
    z = SCD(N, llr_values, melted)

    # Returns original src_vector
    decoded_src = decode_codeword(z, N, melted)

    # Reconstructing original message from decoded codeword
    d_msg = [0] * k
    for i in range(N-k, N):
        d_msg[i - (N-k)] = decoded_src[Q[i]]


    max_label_len = 36
    print("---------------------------------------Global Parameters---------------------------------------")
    print(f"{'N: '.ljust(max_label_len)}", N)
    print(f"{'k: '.ljust(max_label_len)}", k)
    print(f"{'SNR_dB: '.ljust(max_label_len)}", SNR_dB)

    print("---------------------------------------Setup---------------------------------------")
    print(f"{'Subsequence Q: '.ljust(max_label_len)}", Q)
    print(f"{'Subsequence F: '.ljust(max_label_len)}", F)
    
    print(f"{'msg: '.ljust(max_label_len)}", "".join(str(x) for x in msg), end="\n\n")

    print("---------------------------------------Encoding---------------------------------------")
    print(f"{'Source vector u: '.ljust(max_label_len)}", u, end="\n\n")
    print(f"{'Encoded vector y: '.ljust(max_label_len)}", y, end="\n\n")
    print(f"{'Brute force Encoding y_prime: '.ljust(max_label_len)}", y_brute, end="\n\n")
    print(f"{'Encoding correct? (y == y_prime): '.ljust(max_label_len)}", y == y_brute, end="\n\n")
    
    print("---------------------------------------Channel Transmission---------------------------------------")
    print(f"{'Used Indices: '.ljust(max_label_len)}", melted, end="\n\n")
    print(f"{'BPSK symbols: '.ljust(max_label_len)}", bpsk_symbols, end="\n\n")
    print(f"{'Noisy symbols: '.ljust(max_label_len)}", noisy_symbols, end="\n\n")
    print(f"{'LLR values: '.ljust(max_label_len)}", llr_values, end="\n\n")
    
    print("---------------------------------------Decoding---------------------------------------")
    print(f"{'Decoded noisy symbols: '.ljust(max_label_len)}", z, end="\n\n")
    print(f"{'Decoding correct? ( y == z)?: '.ljust(max_label_len)}", z == y, end="\n\n")
    
    print("---------------------------------------Reconstruction------------------------------------------")
    print(f"{'Decoded src_vector: '.ljust(max_label_len)}", decoded_src, end="\n\n")
    print(f"{'msg: '.ljust(max_label_len)}", "".join(str(x) for x in d_msg), end="\n\n")
    print(f"{'Final message == original message?: '.ljust(max_label_len)}", d_msg == msg, end="\n\n")

main()


# def test(input_file, output_file):
#     # Read input parameters from file
#     for (inp, out) in zip(input_file, output_file):
#         with open(inp, 'r') as f:
#             N = int(f.readline())
#             k = int(f.readline())
#             SNR_dB = float(f.readline())
#             msg = [int(x) for x in f.readline().strip()]

#         # Redirect print output to file
#         with open(out, 'w') as f:
#             # Store original stdout
#             original_stdout = sys.stdout
#             sys.stdout = f

#             Q = [i for i in sequence if i < N]
#             F = Q[:N-k]
        
#             # ---- Encoding --- 
#             u = src_vector(msg, N, k, Q)
#             y = encode(u, N, k, Q)
#             y_brute = encoding_check(u, N)

#             # ---- Channel Transmission ----
#             bpsk_symbols = BPSK(y) 
#             noisy_symbols, sigma_2 = AWGN(bpsk_symbols, SNR_dB)
#             llr_values = LLR(noisy_symbols, sigma_2)
#             melted = src_vector([1] * k, N, k, Q)

#             # ---- Decoding ----
#             z = SCD(N, llr_values, melted)
#             decoded_src = decode_codeword(z, N, melted)

#             # Reconstructing original message
#             d_msg = [0] * k
#             for i in range(N-k, N):
#                 d_msg[i - (N-k)] = decoded_src[Q[i]]

#             # Print results (same as in main())
#             max_label_len = 36

#             max_label_len = 36
#             print("---------------------------------------Global Parameters---------------------------------------")
#             print(f"{'N: '.ljust(max_label_len)}", N)
#             print(f"{'k: '.ljust(max_label_len)}", k)
#             print(f"{'SNR_dB: '.ljust(max_label_len)}", SNR_dB)

#             print("---------------------------------------Setup---------------------------------------")
#             print(f"{'Subsequence Q: '.ljust(max_label_len)}", Q)
#             print(f"{'Subsequence F: '.ljust(max_label_len)}", F)
            
#             print(f"{'msg: '.ljust(max_label_len)}", "".join(str(x) for x in msg), end="\n\n")

#             print("---------------------------------------Encoding---------------------------------------")
#             print(f"{'Source vector u: '.ljust(max_label_len)}", u, end="\n\n")
#             print(f"{'Encoded vector y: '.ljust(max_label_len)}", y, end="\n\n")
#             print(f"{'Brute force Encoding y_prime: '.ljust(max_label_len)}", y_brute, end="\n\n")
#             print(f"{'Encoding correct? (y == y_prime): '.ljust(max_label_len)}", y == y_brute, end="\n\n")
            
#             print("---------------------------------------Channel Transmission---------------------------------------")
#             print(f"{'Used Indices: '.ljust(max_label_len)}", melted, end="\n\n")
#             print(f"{'BPSK symbols: '.ljust(max_label_len)}", bpsk_symbols, end="\n\n")
#             print(f"{'Noisy symbols: '.ljust(max_label_len)}", noisy_symbols, end="\n\n")
#             print(f"{'LLR values: '.ljust(max_label_len)}", llr_values, end="\n\n")
            
#             print("---------------------------------------Decoding---------------------------------------")
#             print(f"{'Decoded noisy symbols: '.ljust(max_label_len)}", z, end="\n\n")
#             print(f"{'Decoding correct? ( y == z)?: '.ljust(max_label_len)}", z == y, end="\n\n")
            
#             print("---------------------------------------Reconstruction------------------------------------------")
#             print(f"{'Decoded src_vector: '.ljust(max_label_len)}", decoded_src, end="\n\n")
#             print(f"{'msg: '.ljust(max_label_len)}", "".join(str(x) for x in d_msg), end="\n\n")
#             print(f"{'Final message == original message?: '.ljust(max_label_len)}", d_msg == msg, end="\n\n")
#             sys.stdout = original_stdout


# inp_files = [f"test_in_{i}.txt" for i in range(9)]
# out_files = [f"test_out_{i}.txt" for i in range(9)]

# test(inp_files, out_files)