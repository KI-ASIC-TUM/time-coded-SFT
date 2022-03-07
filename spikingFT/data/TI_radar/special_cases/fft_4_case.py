#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

def main():
    n1 = 1024
    n2 = 512

    data = np.load("data_tum.npy") #(4,1024), 4 cases, 1024 samples per case
    description = { 
            0:"case 1: one strong reflection and one weak reflection@25,201",
            1:"case 2:one weak reflection far away@404",
            2:"two reflections close to each other@42,66",
            3:"multiple reflections@1,39,87,127"
    }
    result = np.zeros((4,n1),dtype=np.complex64)
    for i, chirp_i in enumerate(data):
        fft_cpx = np.fft.fft(chirp_i)
        result[i] = fft_cpx
        fft_p2 = np.abs(fft_cpx)/n1 #two-sided spectrum
        fft_p1 = fft_p2[:n2]*2      #single-sided spectrum
        plt.plot(range(n2),fft_p1)
        plt.xlabel("fft bins")
        plt.ylabel("fft response")
        plt.title(description[i])
        plt.show()
    np.save("fft_result.npy",result)


if __name__ == "__main__":
    main()
