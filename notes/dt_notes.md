# TASK: Upgrade NeurIPS Paper "Cross-Modal Consistency Firewalls"

You are a top-tier Machine Learning theoretician and LaTeX editor. Your task is to upgrade the theoretical rigor of the manuscript by adding dual-sided security guarantees to complement the existing False Positive Rate bounds. 

Please implement the following exact changes to the LaTeX source code:

## 1. Title & Abstract Update
*   **Change the title to:** `Conformal Cross-Modal Firewalls: Dual Bounds on False Positives and Adversarial Evasion`
*   **Update the Abstract:** Inject a sentence stating: *"Furthermore, we prove dual theoretical guarantees: while conformal prediction rigorously bounds the clean-image false positive rate, concentration of measure on the hypersphere provides an exponential lower bound on the adversarial detection rate."*

## 2. Add Section 4.3: Theoretical Bounds on Adversarial Evasion
**Context for Agent:** We need to prove that the firewall catches attacks (bounding the False Negative Rate). 

**Insert Theorem: Geometric Evasion Bound**
Use Concentration of Measure on the unit hypersphere $\mathbb{S}^{d-1}$ to formulate this theorem.
*   **Assumption:** Assume a black-box adversary crafts an injected instruction $s_{\text{adv}}$ that retains malicious semantics (i.e., its alignment with the reference bank is at least $\beta$: $s_{\text{inst}}(s_{\text{adv}}) \ge \beta$). In a transfer-attack setting, the attacker does not know the exact latent embedding $e_x$ of the target image, so the inner product $\langle e_{s_{\text{adv}}}, e_x \rangle$ follows the marginal distribution of a random 1D projection on $\mathbb{S}^{d-1}$.
*   **State the Theorem:** For a calibrated threshold $\tau > 0$, the probability that this attack successfully evades the firewall (i.e., $\Pr[r(s_{\text{adv}}, x) < \tau]$) is bounded by:
    $$ \Pr\big[r(s_{\text{adv}}, x) < \tau\big] \le \exp\left( - \frac{d}{2} \left( \frac{\beta - \tau}{\lambda} \right)_+^2 \right) $$
    where $(z)_+ = \max(0, z)$.
*   **Write the Proof:** Evasion requires $r(s_{\text{adv}}, x) < \tau \implies \lambda \langle e_{s_{\text{adv}}}, e_x \rangle > \beta - \tau$. By Lévy's Lemma on the hypersphere, the probability that the inner product of a fixed vector and a random uniform vector on $\mathbb{S}^{d-1}$ exceeds $t$ decays as $\exp(-dt^2/2)$. Substituting $t = (\beta - \tau)/\lambda$ yields the bound.
*   **Implication (Write in text):** Explain that because the SigLIP embedding dimension $d$ is large, the probability of an attacker blindly guessing an instruction that perfectly mimics the visual context (to lower the inconsistency score) decays exponentially. This provides a mathematically guaranteed detection lower bound.

## 3. Add Section 4.4: Certified Robustness to Adversarial Perturbations
**Context for Agent:** Attackers use typos/leetspeak or pixel noise to evade blocklists. We must prove our risk score is mathematically robust to this.

**Insert Theorem: Lipschitz Continuity of the Risk Score**
*   **Assumption:** Let $\tilde{s}_{\text{adv}}$ be a typographically perturbed version of $s_{\text{adv}}$ such that the text encoder shift is bounded by an $\ell_2$ norm: $\|e_{\tilde{s}_{\text{adv}}} - e_{s_{\text{adv}}}\|_2 \le \epsilon$.
*   **State the Theorem:** The change in the risk score is strictly bounded: 
    $$ |r(\tilde{s}_{\text{adv}}, x) - r(s_{\text{adv}}, x)| \le (1 + \lambda)\epsilon $$
*   **Write the Proof:** Apply the Cauchy-Schwarz inequality to the dot products in the $s_{\text{inst}}$ and $s_{\text{inc}}$ components. Because the embeddings are $L_2$-normalized, the maximum gradient magnitude of the linear combination with respect to the text embedding is bounded by $(1 + \lambda)$.
*   **Implication (Write in text):** Conclude that small adversarial noise (typographical or pixel-level), designed to instantly break keyword blocklists, can only degrade the CMC risk score by a strictly bounded factor $\mathcal{O}(\epsilon)$. This prevents catastrophic defense collapse and guarantees a "Certified Evasion Radius."

## 4. Addressing Covariate Shift (Appendix C)
**Context for Agent:** Standard conformal prediction strictly assumes i.i.d. exchangeability. In a security setting, attackers induce distribution shifts.
**Action:** Add an Appendix titled "Robustness to Adversarial Covariate Shift". Briefly state that if an attacker shifts the image distribution from $P$ to $Q$, standard ICP breaks. However, by adopting **Weighted Conformal Prediction** (Tibshirani et al., 2019) and utilizing the density ratio $w(x) = dQ(x)/dP(x)$ to weight the empirical CDF, our mathematical FPR bound perfectly translates to distribution-shifted adversarial domains.

## 5. Formatting Check
Ensure all math is wrapped properly in `amsthm` environments (`\begin{theorem}`, `\begin{proof}`). Maintain the authoritative, highly analytical academic tone established in previous sections.