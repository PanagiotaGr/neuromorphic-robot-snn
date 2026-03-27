# 📄 Neuromorphic Control of Embodied Agents

## A Comparative Study of Spiking and Artificial Neural Networks

---

## Abstract

Spiking Neural Networks (SNNs) have emerged as a biologically inspired alternative to conventional Artificial Neural Networks (ANNs), offering intrinsic temporal dynamics and event-driven computation. Despite their theoretical advantages, their practical benefits in control systems remain underexplored.

In this work, we investigate the performance of SNNs in a closed-loop robotic control task, comparing them against standard feedforward ANNs. We construct a simulated embodied agent tasked with trajectory tracking under partial observability and evaluate performance under various perturbations, including sensor noise, delay, and degradation.

Our results demonstrate that SNNs, while exhibiting lower supervised learning accuracy, can achieve superior robustness and stability in dynamical environments. This highlights a fundamental mismatch between offline evaluation metrics and real-world control performance, emphasizing the importance of temporal computation in embodied intelligence.

---

## 1. Introduction

Modern machine learning systems are predominantly based on deep artificial neural networks trained using gradient-based optimization. These models are highly effective at approximating static mappings between inputs and outputs. However, real-world agents operate in environments that evolve over time, where decisions must account for temporal dependencies and delayed feedback.

Spiking Neural Networks (SNNs) differ fundamentally from ANNs by incorporating time as a first-class computational dimension. Instead of continuous activations, SNNs communicate via discrete spike events and maintain internal state through membrane potentials. This enables them to naturally encode temporal information and operate in event-driven settings.

The central hypothesis of this work is:

> Temporal dynamics in SNNs provide advantages in closed-loop control tasks, particularly under uncertainty, noise, and delayed observations.

---

## 2. Problem Formulation

We consider a robot navigating a two-dimensional track using only local sensor observations.

### System Definition

```
State:        s_t = (x_t, y_t, θ_t)
Observation:  o_t ∈ ℝⁿ
Action:       a_t ∈ [-1, 1]   (continuous steering)
```

The system is **partially observable**, as the agent does not have access to global position or full environment information.

### Objective

```
minimize   lateral deviation from track centerline
maximize   forward progress
```

This defines a continuous control problem under uncertainty.

---

## 3. Methods

## 3.1 Artificial Neural Network (ANN)

The ANN controller is a feedforward network:

```
f_θ: ℝⁿ → ℝ
```

* ReLU activations
* Mean Squared Error (MSE) loss
* No explicit temporal state

This model approximates a static mapping:

```
o_t → a_t
```

---

## 3.2 Spiking Neural Network (SNN)

### LIF Neuron Dynamics

```
tau_m * dV/dt = -V + I
```

### Spike Condition

```
V ≥ V_th  → spike
```

The membrane potential integrates input over time, providing **implicit memory**.

---

### Temporal Processing

The SNN processes each input across multiple time steps:

```
o_t → {s₁, s₂, ..., s_T}
```

### Output Readout

```
a_t = tanh(W * s_mean)
```

where:

```
s_mean = average spike activity over time
```

Training is performed using **surrogate gradient methods**, allowing backpropagation through non-differentiable spike events.

---

## 3.3 Spike Encoding

Continuous inputs must be transformed into spike trains.

### Rate Coding

```
spike probability ∝ input magnitude
```

### Latency Coding

```
strong input → early spike
weak input   → late spike
```

### Population Coding

```
input → distributed representation across neuron populations
```

---

## 3.4 Continuous Control

Unlike classification-based control, we define a continuous action space:

```
a_t ∈ [-1, 1]
```

This formulation better reflects real-world robotic systems, where control signals are inherently continuous.

---

## 4. Experimental Setup

### Dataset

* Procedurally generated tracks
* Randomized robot states
* Teacher controller provides supervision

### Perturbations

We evaluate robustness under:

```
• Gaussian noise
• Temporal delay
• Sensor dropout
• Dead sensors
```

### Metrics

```
success_rate
mean_lateral_error
max_lateral_error
episode_length
control_smoothness
spike_activity (SNN)
```

---

## 5. Results

## 5.1 Offline vs Closed-loop Performance

```
ANN → higher supervised accuracy
SNN → better control stability
```

This reveals a fundamental issue:

```
offline accuracy ≠ closed-loop performance
```

---

## 5.2 Robustness

SNNs demonstrate:

```
✓ improved tolerance to delay
✓ stability under noise
✓ graceful degradation under sensor failure
```

This is attributed to temporal integration and internal state dynamics.

---

## 5.3 Encoding Analysis

Different encoding strategies significantly affect performance:

```
Rate coding       → stable baseline
Latency coding    → high sensitivity
Population coding → robust but higher complexity
```

---

## 5.4 Continuous Control Behavior

```
ANN → smoother initial outputs
SNN → adaptive temporal smoothing
```

Under perturbations, SNNs maintain more consistent trajectories.

---

## 6. Discussion

Traditional evaluation methods in machine learning emphasize static metrics such as classification accuracy. However, these metrics fail to capture the behavior of systems interacting with dynamic environments.

Control problems require:

```
state evolution + temporal consistency
```

SNNs inherently provide:

```
• temporal integration
• implicit memory
• event-driven computation
```

Despite these advantages, challenges remain:

```
• training instability
• sensitivity to hyperparameters
• limited software tooling
```

---

## 7. Conclusion

We demonstrate that:

```
SNNs can outperform ANNs in embodied control tasks
despite inferior offline performance metrics
```

This supports a broader perspective:

```
intelligence emerges from interaction over time
not just static input-output mappings
```

---

## 8. Future Work

* Reinforcement learning with SNN policies
* Deployment on neuromorphic hardware
* Event-based vision integration
* Multi-agent coordination
* Energy-aware evaluation

---

## 9. Reproducibility

```
python main.py
python experiments_mode.py
python encoding_study.py
python continuous_steering.py
python continuous_benchmark.py
```

---

## 10. Keywords

```
Spiking Neural Networks
Neuromorphic Computing
Robotics
Control Systems
Temporal Dynamics
Embodied AI
```

---

## Author
Panagiota Grosdouli

## Citation

If you use this work for research, please cite it:

```bibtex
@software{Grosdouli_SNN_Robotics_2026,
  author = {Grosdouli, Panagiota},
  title = {Neuromorphic Control of Embodied Agents: A Comparative Study of Spiking and Artificial Neural Networks},
  year = {2026},
  url = {[https://github.com/PanagiotaGr/neuromorphic-robot-snn](https://github.com/PanagiotaGr/neuromorphic-robot-snn)}
}

