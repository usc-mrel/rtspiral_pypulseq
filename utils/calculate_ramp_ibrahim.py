import numpy as np

def calculate_ramp_ibrahim(nTRs, T1, T2, TR, alpha, max_alpha=np.pi):
    # calculate bSSFP ramp for RF pulses from the following paper:
    # Improved Myocardial Tagging Contrast in Cine Balanced SSFP Images
    # by El-Sayed H. Ibrahim et al. JCMR 2006

    Bs = [alpha / 2]

    E1 = np.exp(-TR / T1)
    E2 = np.exp(-TR / T2)

    for i in range(1, nTRs + 1):
        Bprev = Bs[-1]
        Bn = np.arcsin(np.sin(Bprev) / ((E1 * np.cos(Bprev) * np.cos(Bprev)) + (E2 * np.sin(Bprev) * np.sin(Bprev))))
        Bs.append(Bn)

    As = [Bs[i] + Bs[i - 1] for i in range(1, nTRs)]

    # check if the maximum alpha is exceeded, and if so, clip it
    # if any A is NaN set it to max_alpha
    As = [max_alpha if np.isnan(As[i]) else As[i] for i in range(0, len(As))]
    As = np.clip(As, 0, max_alpha)

    return As