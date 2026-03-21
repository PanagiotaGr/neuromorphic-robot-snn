# Neuromorphic Control of Embodied Agents:

## A Comparative Study of Spiking and Artificial Neural Networks

---

## Abstract

This project investigates the use of **Spiking Neural Networks (SNNs)** for control in embodied robotic systems, in comparison to conventional **Artificial Neural Networks (ANNs)**. While ANNs typically excel in supervised learning benchmarks, their performance does not always translate to stable behavior in closed-loop environments.

We construct a simulated robotic agent tasked with trajectory tracking using only local sensory input, and systematically evaluate both ANN and SNN controllers under varying environmental perturbations. Additionally, we explore the role of **temporal spike encoding**, **continuous control**, and **sensor degradation**.

Our findings suggest that SNNs, despite lower offline accuracy, can exhibit superior robustness and stability in dynamical settings, highlighting the importance of temporal computation in control systems.

---

## 1. Introduction

The dominant paradigm in modern machine learning relies on deep artificial neural networks trained via backpropagation. However, such models are inherently **static function approximators**, lacking explicit mechanisms for temporal state representation beyond architectural extensions (e.g., RNNs).

In contrast, **Spiking Neural Networks (SNNs)** operate using discrete events in time and maintain internal dynamics through membrane potentials, making them inherently suited for **time-dependent processing** and **event-driven environments**.

This project explores the hypothesis:

> *Temporal dynamics in SNNs provide advantages in closed-loop control tasks, particularly under uncertainty and delayed feedback.*

---

## 2. Problem Formulation

We consider a robot navigating a 2D procedurally generated path. The system is defined as a **partially observable dynamical system**:

* State: ( s_t = (x_t, y_t, \theta_t) )
* Observation: ( o_t \in \mathbb{R}^n ) (sensor readings)
* Action:

  * discrete: ( a_t \in {left, forward, right} )
  * continuous: ( a_t \in [-1, 1] )

The control objective is to minimize deviation from the track centerline while progressing forward.

---

## 3. Methods

### 3.1 Artificial Neural Network (ANN)

A feedforward network:

[
f_{\theta}: \mathbb{R}^n \rightarrow \mathbb{R}^k
]

* ReLU activations
* Cross-entropy loss (discrete) or MSE (continuous)
* No internal temporal state

---

### 3.2 Spiking Neural Network (SNN)

We employ **Leaky Integrate-and-Fire (LIF)** neurons:

[
\tau_m \frac{dV(t)}{dt} = -V(t) + I(t)
]

A spike is emitted when:
[
V(t) \geq V_{th}
]

Training is performed using **surrogate gradients**, enabling backpropagation through non-differentiable spike events.

The network processes inputs over ( T ) discrete time steps:

[
o_t \rightarrow {s_t^{(1)}, s_t^{(2)}, ..., s_t^{(T)}}
]

Output is obtained via temporal aggregation:
[
\hat{y} = \sum_{t=1}^{T} s_t
]

---

### 3.3 Spike Encoding

We investigate three encoding strategies:

* **Rate Coding**: spike probability proportional to input magnitude
* **Latency Coding**: spike timing encodes value
* **Population Coding**: distributed representation across neuron groups

These affect how continuous sensor signals are transformed into spike trains.

---

### 3.4 Continuous Control Extension

We extend the action space to:

[
a_t \in [-1, 1]
]

representing steering angle.

For SNNs, regression is performed via a readout layer over accumulated spike activity:

[
a_t = \tanh(W \cdot \bar{s})
]

where ( \bar{s} ) is the average spike activity over time.

---

## 4. Experimental Setup

### Dataset

* Procedurally generated tracks
* Randomized robot states
* Teacher controller provides supervision

### Evaluation Conditions

* Gaussian sensor noise
* Temporal delay
* Sensor dropout
* Sensor failure (dead channels)

### Metrics

* Success rate
* Mean lateral error
* Maximum deviation
* Episode duration
* Control smoothness
* Spike activity (SNN only)

---

## 5. Results and Analysis

### 5.1 Offline vs Closed-loop Performance

We observe a consistent mismatch:

* ANN achieves higher supervised accuracy
* SNN demonstrates better stability in simulation

This indicates that **classification accuracy is not a sufficient proxy** for control quality in dynamical systems.

---

### 5.2 Robustness

SNNs exhibit:

* Improved tolerance to delayed observations
* Graceful degradation under sensor corruption
* More stable trajectories under noise

This behavior is attributed to internal temporal integration.

---

### 5.3 Encoding Effects

Performance varies significantly across encoding methods:

* Rate coding: stable baseline
* Latency coding: sensitive but expressive
* Population coding: robust but higher dimensional

This confirms that **input representation is critical** in SNN systems.

---

### 5.4 Continuous Control

In continuous steering:

* ANN produces smoother outputs initially
* SNN develops adaptive temporal smoothing
* Differences become more pronounced under noise

---

## 6. Discussion

This study highlights key limitations of standard evaluation practices:

* Static metrics (accuracy) fail to capture **closed-loop dynamics**
* Temporal computation emerges as a crucial factor in control

SNNs provide:

* implicit memory through membrane dynamics
* event-driven computation
* robustness in non-ideal conditions

However, challenges remain:

* difficult training dynamics
* sensitivity to hyperparameters
* lack of standardized tools

---

## 7. Conclusion

We demonstrate that:

> SNNs can outperform ANNs in embodied control tasks despite inferior offline metrics.

This supports the broader view that:

* intelligence in agents is not purely a function of static input-output mappings
* but emerges from **interaction over time**

---

## 8. Future Work

* Reinforcement learning with SNN policies
* Neuromorphic hardware deployment
* Event-based vision integration
* Multi-agent coordination
* Energy-aware evaluation

---

## 9. Reproducibility

All experiments can be reproduced via:

```bash
python main.py
python experiments_mode.py
python encoding_study.py
python continuous_steering.py
python continuous_benchmark.py
```

---

## 10. Keywords

Spiking Neural Networks, Neuromorphic Computing, Robotics, Control Systems, Temporal Dynamics, Embodied AI

---

## Author

Student project in:

* Machine Learning
* Robotics
* Neuromorphic Systems

---

> This project serves as a bridge between machine learning and dynamical systems, emphasizing the importance of time in intelligent behavior.
