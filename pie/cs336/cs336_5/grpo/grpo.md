# 策略梯度




显然，策略梯度可以写成
$$
\nabla_{\theta}J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^T \nabla_{\theta}\log \pi_\theta(a_t | s_t)R(\tau)]
$$


给定策略$\pi_\theta$，就可以在所有路径构成的集合$T$上定义一个采样分布。要计算策略的梯度$\nabla_{\theta}J(\theta)$，需从该分布中采样若干条路径，对每一条路径$\tau$，得到对应梯度，之后被$R(\tau)$修正后求和即可。
为了计算这些梯度，可以使用代理损失:

$$
\mathcal{L} = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^T \frac{\pi_\theta(a_t | s_t)}{sg(\pi_\theta(a_t | s_t))} R(\tau)]
$$
其中sg是阻止梯度回传的算子。可以证明该函数$\mathcal{L}$的梯度就是我们要算的策略的梯度。由于$sg(\pi_\theta(a_t | s_t))$与$R(\tau)$不提供梯度，此时有：
$$
\nabla_\theta\mathcal{L} = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^T  \frac{\nabla_\theta \pi_\theta(a_t | s_t)}{sg(\pi_\theta(a_t | s_t))} R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^T \nabla_{\theta}\log \pi_\theta(a_t | s_t)R(\tau)] = \nabla_{\theta}J(\theta)
$$
可以看到只要定义损失$\mathcal{L}$，再反向传播即可求得策略梯度。进一步地，为了减少$\mathcal{L}$的方差，设$b(s_t)$是一个只依赖状态$s_t$与$\pi_\theta$的函数，计算如下梯度：
$$
B = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^T \nabla_{\theta}\log \pi_\theta(a_t | s_t)(R(\tau)-b(s_t))]
$$
可以证明$B$与$\nabla_{\theta}J(\theta)$是相等的。使用如上损失进行方向传播,减少方差的同时还是无偏的,反向传播后得到的梯度即为策略梯度(即使这个损失没什么实际意义,该损失的唯一作用是在反向传播的时候恰好的计算策略梯度),随后更新模型，再进行进一步采样，计算梯度，如此下去完成训练。


以上方式是在线的.可以使用非在线的方法,继续使用来自先前老策略$\pi_{\theta_{old}}$采样的$\tau$,只需要使用重采样将$\mathcal{L}$修改为如下即可:

$$
\mathcal{L} = \mathbb{E}_{\tau \sim \pi_{\theta_{old}}}[\sum_{t=0}^T \frac{\pi_\theta(a_t | s_t)}{sg(\pi_{\theta_{old}}(a_t | s_t))} R(\tau)]
$$
文献指出如上修改是一种近似,此处是我不太理解的地方.感觉是严格相等.

# GRPO

## Algorithm (Single Outer Iteration)

### Step 1: Snapshot

$\pi_{\theta_{old}} \leftarrow \pi_\theta$

### Step 2: Experience Collection

1. Sample a batch of prompts $\mathcal{Q} = \{q_1, \dots, q_B\}$ from the dataset.
2. **For each** $q \in \mathcal{Q}$:
   - Rollout $G$ responses $\{o_1, \dots, o_G\}$ by sampling from $\pi_{\theta_{\text{old}}}(\cdot \mid q)$.
   - For each response $o_g, 1 \leq g \leq G$:
     - Compute reward $r_g$ then advantage $A_g$
   - Store experience tuple $E = (q, o_g, r_g, A_g)$.
   - Store logits of the response's tokens (Grad must not be presented, uses detach()).

**Padding & Masking** (as per the described scheme):
- Prompts $q$ are **left‑padded** to a fixed length before sending to the inference engine.
- Responses $o_g$ are **right‑padded** to a fixed length after collecting the response from the inference engine.
- We concatenat prompt + response to obtain the tesnor. It has mask $\mathcal{M}$ where $\mathcal{M}[t] = 1$ iff token $t$ belongs to the response part or prompt part.
- Also antoher mask $\mathcal{M}_{\text{resp}}[t] = 1$ iff token $t$ belongs to the response part.
- Each experience tuple $(q, o_g, r_g, A_g)$ has its own $\mathcal{M}$ and $\mathcal{M}_{\text{resp}}$.

---

### Step 3: Policy Update Uses Old Experiences (Inner Epochs)

For $k = 1$ **to** $K$:

1. (Optional) Shuffle the collected experiences.
2. **For each** iteration:
   - **Forward pass (teacher forcing):**  
     Concatenated prompt and response tokens are feed through $\pi_\theta$, which yield logits. (Use $\mathcal{M}$ for causal attention).
   - **Per‑token probability ratio** (for each response token $t$):
     Get ratio: $
     \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)},
     $
   - **Per token clipped surrogate objective**:
     Calculate clipped surrogate objective for each token by ratio and advantages. Both advantages and ratio's denominator are leaves in the computational graph and cannot carry gradients.
   - **Loss** (negative, averaged over response tokens):
     Calculate loss $\mathcal{L}$ over **response tokens** by  $\mathcal{M}_{\text{resp}}$, whose value is meaningless and backward will populate correct gradients of our policy. 
   - Compute $\nabla_\theta \mathcal{L}$ and update $\theta$.

---

### Step 4: Iterate

Go to **Step 1** (snapshot policy again) and repeat.

---
