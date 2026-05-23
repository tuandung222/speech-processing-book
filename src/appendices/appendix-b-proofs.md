# Phụ lục B: Mathematical Proofs

## Mục đích

Phụ lục B chứa các chứng minh toán học cho các công thức quan trọng xuất hiện trong các chương chính. Trong các chương, công thức được phát biểu và sử dụng; chứng minh chi tiết được chuyển về Phụ lục B để giữ mạch đọc chính không bị gián đoạn. Người đọc cần thực sự hiểu sâu (viết paper, giải thích cho team) nên tham khảo phụ lục này; người đọc thực hành có thể bỏ qua.

Các chứng minh hiện có: DFT completeness, CTC forward-backward derivation, tính chất triangular của mel filterbank.

## Discrete Fourier Transform (DFT) Completeness

### Statement

DFT basis vectors form an orthogonal set. Any discrete signal of length $N$ can be perfectly reconstructed from its $N$ DFT coefficients.

### Proof

**Orthogonality of DFT basis vectors:**

Define DFT basis vector $\mathbf{w}_k = [1, e^{-j2\pi k/N}, e^{-j2\pi k \cdot 2/N}, \ldots, e^{-j2\pi k(N-1)/N}]$.

<a id="eq-dft-orthogonality"></a>

$$
\langle \mathbf{w}_k, \mathbf{w}_l \rangle = \sum_{n=0}^{N-1} e^{-j2\pi kn/N} \cdot e^{j2\pi ln/N} = \sum_{n=0}^{N-1} e^{j2\pi(l-k)n/N}
$$

**Case 1:** $k = l$

<a id="eq-idft-proof"></a>

$$
\sum_{n=0}^{N-1} e^{j2\pi \cdot 0 \cdot n/N} = \sum_{n=0}^{N-1} 1 = N
$$

**Case 2:** $k \neq l$, let $r = l - k \neq 0$

$$
\sum_{n=0}^{N-1} e^{j2\pi rn/N} = \frac{1 - e^{j2\pi r}}{1 - e^{j2\pi r/N}} = \frac{1 - 1}{1 - e^{j2\pi r/N}} = 0
$$

since $e^{j2\pi r} = 1$ for integer $r$.

Therefore: $\langle \mathbf{w}_k, \mathbf{w}_l \rangle = N \cdot \delta_{kl}$  -  orthogonal basis. $\blacksquare$

**Inverse DFT follows directly:**

$$
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \cdot e^{j2\pi kn/N}
$$

## CTC Forward-Backward Algorithm

### Statement

The CTC loss can be efficiently computed in $O(T \times U)$ time using dynamic programming, where $T$ is the number of frames and $U$ is the label length.

### Forward Variable

Define $\alpha_t(s)$ = probability of outputting the first $s$ symbols of the modified label sequence $\ell'$ (with blanks) in $t$ time steps:

<a id="eq-ctc-forward-def"></a>

$$
\alpha_t(s) = P(\pi_{1:t} \text{ maps to } \ell'_{1:s} \mid \mathbf{x})
$$

**Initialization** ($t = 1$):

<a id="eq-ctc-forward-init"></a>

$$
\alpha_1(1) = P(\text{blank} \mid \mathbf{x}_1), \quad \alpha_1(2) = P(\ell_1 \mid \mathbf{x}_1), \quad \alpha_1(s) = 0 \text{ for } s > 2
$$

**Recursion** ($t > 1$):

<a id="eq-ctc-forward-recursion"></a>

$$
\alpha_t(s) = P(\ell'_s \mid \mathbf{x}_t) \cdot \begin{cases}
\alpha_{t-1}(s) + \alpha_{t-1}(s-1) & \text{if } \ell'_s = \text{blank or } \ell'_s = \ell'_{s-2} \\
\alpha_{t-1}(s) + \alpha_{t-1}(s-1) + \alpha_{t-1}(s-2) & \text{otherwise}
\end{cases}
$$

**Proof of correctness:**

At time $t$, the token at position $s$ in $\ell'$ can be reached from:

1. **Same position** $s$ at $t-1$: Repeating the same token
2. **Previous position** $s-1$ at $t-1$: Moving to next token
3. **Two positions back** $s-2$ at $t-1$: Skipping a blank (only if $\ell'_s \neq \ell'_{s-2}$, i.e., no repeated characters)

Case 3 is excluded when $\ell'_s = \text{blank}$ or when two consecutive non-blank tokens are the same, because skipping the blank between them would cause a merge. $\blacksquare$

### Total CTC Loss

<a id="eq-ctc-total"></a>

$$
P(\ell \mid \mathbf{x}) = \alpha_T(|\ell'|) + \alpha_T(|\ell'| - 1)
$$

The two terms account for paths ending with the last label or with a trailing blank.

<a id="eq-ctc-loss-final"></a>

$$
\mathcal{L}_{\text{CTC}} = -\log P(\ell \mid \mathbf{x})
$$

## VQ-VAE ELBO Derivation

### Statement

The VQ-VAE training objective is a lower bound on the log-likelihood $\log p_\theta(\mathbf{x})$.

### Derivation

Starting from the marginal log-likelihood:

<a id="eq-vqvae-marginal"></a>

$$
\log p_\theta(\mathbf{x}) = \log \sum_{\mathbf{z}} p_\theta(\mathbf{x}, \mathbf{z})
$$

Introduce approximate posterior $q_\phi(\mathbf{z} \mid \mathbf{x})$:

<a id="eq-vqvae-importance"></a>

$$
\log p_\theta(\mathbf{x}) = \log \sum_{\mathbf{z}} q_\phi(\mathbf{z} \mid \mathbf{x}) \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})}
$$

By Jensen's inequality ($\log$ is concave):

<a id="eq-vqvae-elbo-derivation"></a>

$$
\log p_\theta(\mathbf{x}) \geq \sum_{\mathbf{z}} q_\phi(\mathbf{z} \mid \mathbf{x}) \log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})} = \text{ELBO}
$$

Expanding:

<a id="eq-vqvae-elbo-expanded"></a>

$$
\text{ELBO} = \underbrace{\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x} \mid \mathbf{z})]}_{\text{reconstruction}} - \underbrace{D_{\text{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z}))}_{\text{regularization}}
$$

**For VQ-VAE specifically:**

The posterior is **deterministic** (argmin quantization): $q_\phi(\mathbf{z} = \mathbf{e}_k \mid \mathbf{x}) = \mathbf{1}[k = \arg\min_j \|\text{enc}(\mathbf{x}) - \mathbf{e}_j\|]$

With a uniform prior $p(\mathbf{z}) = 1/K$ over $K$ codebook entries:

<a id="eq-vqvae-kl"></a>

$$
D_{\text{KL}} = \log K - H(q_\phi) = \log K - 0 = \log K \quad (\text{constant})
$$

Since the KL term is constant, maximizing ELBO reduces to minimizing reconstruction loss + keeping encoder close to codebook entries:

<a id="eq-vqvae-loss-derived"></a>

$$
\mathcal{L}_{\text{VQ-VAE}} = \|\mathbf{x} - \text{dec}(\mathbf{e}_{k^*})\|^2 + \|\text{sg}(\text{enc}(\mathbf{x})) - \mathbf{e}_{k^*}\|^2 + \beta\|\text{enc}(\mathbf{x}) - \text{sg}(\mathbf{e}_{k^*})\|^2
$$

$\blacksquare$

## Conditional Flow Matching Derivation

### Statement

The Conditional Flow Matching (CFM) objective provides a simulation-free training procedure for continuous normalizing flows.

### Setup

We want to learn a time-dependent velocity field $v_\theta(\mathbf{x}, t)$ that transforms a simple distribution $p_0$ (e.g., Gaussian) into a target distribution $p_1$ (e.g., data) via the ODE:

<a id="eq-cfm-ode"></a>

$$
\frac{d\mathbf{x}_t}{dt} = v_\theta(\mathbf{x}_t, t), \quad t \in [0, 1]
$$

### Optimal Transport Path

Choose the **linear interpolation path** (optimal transport):

<a id="eq-cfm-ot-path"></a>

$$
\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1, \quad \mathbf{x}_0 \sim p_0, \quad \mathbf{x}_1 \sim p_1
$$

The conditional velocity field generating this path is:

<a id="eq-cfm-cond-velocity"></a>

$$
u_t(\mathbf{x} \mid \mathbf{x}_1) = \mathbf{x}_1 - \mathbf{x}_0
$$

**Proof:** Taking the time derivative of the interpolation:

<a id="eq-cfm-velocity-proof"></a>

$$
\frac{d\mathbf{x}_t}{dt} = \frac{d}{dt}[(1-t)\mathbf{x}_0 + t\mathbf{x}_1] = -\mathbf{x}_0 + \mathbf{x}_1 = \mathbf{x}_1 - \mathbf{x}_0
$$

### Training Objective

The CFM loss matches the learned velocity to the conditional velocity:

<a id="eq-cfm-loss-derived"></a>

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1], \mathbf{x}_0 \sim p_0, \mathbf{x}_1 \sim p_1} \left[\|v_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0)\|^2\right]
$$

**Key result** [^lipman2023flow]: Minimizing this conditional objective is equivalent to minimizing the **marginal** flow matching objective:

<a id="eq-fm-marginal"></a>

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, \mathbf{x}_t \sim p_t} \left[\|v_\theta(\mathbf{x}_t, t) - u_t(\mathbf{x}_t)\|^2\right]
$$

The proof relies on the fact that $\nabla_\theta \mathcal{L}_{\text{CFM}} = \nabla_\theta \mathcal{L}_{\text{FM}}$ (gradients are equal), making CFM a valid training procedure. $\blacksquare$

## Gumbel-Softmax Gradient Estimator

### Statement

Gumbel-Softmax provides a differentiable approximation to sampling from a categorical distribution, enabling gradient-based optimization through discrete choices.

### Gumbel-Max Trick

To sample from a categorical distribution with logits $\ell_1, \ldots, \ell_K$:

<a id="eq-gumbel-max"></a>

$$
k^* = \arg\max_k [\ell_k + g_k], \quad g_k \sim \text{Gumbel}(0, 1)
$$

where $g_k = -\log(-\log(u_k)), \quad u_k \sim \text{Uniform}(0, 1)$.

### Continuous Relaxation

Replace $\arg\max$ with softmax at temperature $\tau$:

<a id="eq-gumbel-softmax"></a>

$$
y_k = \frac{\exp((\ell_k + g_k) / \tau)}{\sum_{j=1}^{K} \exp((\ell_j + g_j) / \tau)}
$$

**Properties:**

- As $\tau \to 0$: $\mathbf{y}$ approaches one-hot (exact categorical sample)
- As $\tau \to \infty$: $\mathbf{y}$ approaches uniform distribution
- For any $\tau > 0$: $\mathbf{y}$ is differentiable w.r.t. $\ell_k$

**Proof of convergence:**

<a id="eq-gumbel-limit"></a>

$$
\lim_{\tau \to 0} y_k = \begin{cases} 1 & \text{if } k = \arg\max_j (\ell_j + g_j) \\ 0 & \text{otherwise} \end{cases}
$$

This holds because as $\tau \to 0$, softmax concentrates all mass on the maximum element. $\blacksquare$

### Application in Wav2Vec 2.0

Wav2Vec 2.0 uses Gumbel-Softmax to make codebook selection differentiable during training:

<a id="eq-gumbel-w2v"></a>

$$
\mathbf{q} = \sum_{k=1}^{K} y_k \cdot \mathbf{e}_k, \quad y_k = \text{GumbelSoftmax}(\ell_k, \tau)
$$

Temperature $\tau$ is annealed from 2.0 to 0.5 during training.

## ELBO for VITS

### Statement

VITS maximizes a variational lower bound on $\log p_\theta(\mathbf{x} \mid c)$ where $c$ is text conditioning.

### Derivation

<a id="eq-vits-marginal"></a>

$$
\log p_\theta(\mathbf{x} \mid c) = \log \int p_\theta(\mathbf{x} \mid \mathbf{z}) p_\theta(\mathbf{z} \mid c) \, d\mathbf{z}
$$

Introducing variational posterior $q_\phi(\mathbf{z} \mid \mathbf{x})$:

<a id="eq-vits-elbo-proof"></a>

$$
\log p_\theta(\mathbf{x} \mid c) \geq \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x} \mid \mathbf{z})]}_{\text{reconstruction (HiFi-GAN)}} - \underbrace{D_{\text{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p_\theta(\mathbf{z} \mid c))}_{\text{prior matching}}
$$

The normalizing flow in VITS transforms the posterior to make it more flexible:

<a id="eq-vits-flow-posterior"></a>

$$
q_\phi(\mathbf{z}_K \mid \mathbf{x}) = q_\phi(\mathbf{z}_0 \mid \mathbf{x}) \prod_{k=1}^{K} \left|\det \frac{\partial f_k^{-1}}{\partial \mathbf{z}_k}\right|
$$

This allows $q_\phi$ to model complex, multi-modal posterior distributions while maintaining tractable density evaluation. $\blacksquare$



---

<!-- References (auto-generated from .bib) -->
[^lipman2023flow]: Lipman, Yaron and Chen, Ricky T Q and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matthew, "Flow Matching for Generative Modeling", International Conference on Learning Representations
