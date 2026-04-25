# Theorem Verification Results

## Theorem 1: Levy's Lemma (Geometric Evasion Bound)

- t=0.05: empirical tail=4.41e-02, Levy bound=2.37e-01, bound holds=True
- t=0.10: empirical tail=3.35e-04, Levy bound=3.15e-03, bound holds=True
- t=0.15: empirical tail=0.00e+00, Levy bound=2.35e-06, bound holds=True
- t=0.20: empirical tail=0.00e+00, Levy bound=9.86e-11, bound holds=True

## Theorem 2: Lipschitz Continuity

- eps=0.010: max observed |delta r|=0.0009, (1+lambda)*eps=0.0150, bound holds=True
- eps=0.050: max observed |delta r|=0.0051, (1+lambda)*eps=0.0750, bound holds=True
- eps=0.100: max observed |delta r|=0.0119, (1+lambda)*eps=0.1500, bound holds=True
- eps=0.200: max observed |delta r|=0.0308, (1+lambda)*eps=0.3000, bound holds=True

## Theorem 3: DKW Concentration of Conditional FPR

- alpha=0.1, n_cal=100: empirical FPR mean=0.0986, std=0.0322
- Beta-predicted std=0.0300, DKW bound=1.21e+00, DKW holds=True

## vMF Concentration Fit

- kappa_0 (benign->image) = 44.09
- kappa_1 (span->ref bank) = 1319.54
- lambda implied by vMF = 0.033 (config uses 0.500)