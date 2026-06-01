# Auditory Prediction Primacy

## Why Ears Must Understand the World Before Eyes Can Align

**——On the Logical Priority of Unimodal Physical Modeling in Joint Embedding Predictive Architectures**

---

### Abstract

We argue that cross-modal alignment in Joint Embedding Predictive Architectures (JEPA) is fundamentally bounded by the depth of representation learned in the unimodal pretraining phase. Using audio as the primary case study, we demonstrate that a purely acoustic encoder—when trained with temporal prediction objectives, slot-based object disentanglement, and contrastive fine-tuning—can independently acquire representations encoding physical properties such as distance, velocity, spatial occlusion, and multi-source separation. We formalize the principle: *alignment is a pull in representation space, not a transfer of knowledge; its ceiling is set by the independent capacity of each modality to model physical regularities from raw sensory input.*

---

### 1. Introduction: The U-Shaped Hearing Curve as a Biological Prior

The author's audiometric profile exhibits an atypical U-shaped sensitivity curve: hyperacuity below 3 kHz, relative insensitivity in the 3–8 kHz band, and preserved detection above 20 kHz. While most listeners exhibit peak sensitivity in the 3–8 kHz range—corresponding to formant discrimination, fricative perception, and alert-signal detection—the author's sensitivity is biased toward low-frequency resonance (spatial reverberation, source proximity) and high-frequency harmonic transients (air absorption, texture cues).

This deviation from the population mean is a natural experiment: **different cochlear transfer functions extract different semantic primitives from the same physical pressure wave.** Auditory perception is thus not a faithful reproduction of the spectral envelope; it is a task-conditioned, lossy compression of acoustic energy into behaviorally relevant latent variables:

$$\text{Perception}(x) = f_{\text{cochlea}} \circ f_{\text{attention}} \circ f_{\text{prediction}}(x)$$

where $f_{\text{cochlea}}$ is the biological/mel-scale filterbank, $f_{\text{attention}}$ is the learned selection of task-relevant features, and $f_{\text{prediction}}$ is the forward-model that generates expectations about future sensory states. Only the composition of all three yields understanding.

---

### 2. Predictive Coding as the Biological Basis of JEPA

LeCun's Joint Embedding Predictive Architecture posits that *prediction is understanding.* We extend this with a biological corollary: **understanding presupposes selection—only signals with predictive value ascend to the representation space.**

Consider congenital blindness: an individual who has never seen a room can nonetheless judge corridor length, door proximity, and occupant count through audition alone. The auditory system discards spectral detail irrelevant to physical interaction—wall color, floor texture—and retains only acoustic cues covarying with spatial structure: reverberation time, interaural level differences, spectral centroid shift.

This is precisely what JEPA achieves architecturally. Prediction occurs not in raw signal space (pixel, waveform) but in a learned latent space where irrelevant variation has already been compressed away. The encoder—analogous to the cochlea plus cortical filtering—selects what enters the representation. The predictor—analogous to forward-model circuits in auditory cortex—generates expectations. The mismatch drives learning. **No reconstruction loss needed.**

The system learns what matters by learning what changes.

---

### 3. Slot Attention as Object-Centric Physical Modeling

Natural acoustic scenes are polyphonic. A single microphone captures overlapping signals from independent physical sources: rain, train wheels, footsteps, wind. Monolithic encoders collapse all sources into a single entangled latent vector, erasing the very structure that physical reasoning requires.

We adopt **Slot Attention with Object-Level Temporal Masking (C-JEPA).** Given $K = 5$ learnable slot queries per timestep, each receiving a distinct, learnable projection of the shared audio feature vector, the model competitively assigns temporal structure to slots via iterative attention:

$$\text{Slot}_k = \frac{\sum_{i} A_{k,i} \cdot v_i}{\sum_{i} A_{k,i}}, \quad A_{k,i} = \text{softmax}_k\left(\frac{q_k^\top k_i}{\sqrt{d}}\right)$$

The training signal forcing slot differentiation is purely self-supervised: **mask entire object histories and predict their future states.** For each training sample, a random subset of $K$ slots has their full temporal history (all 6 input frames) replaced with a learnable mask token. The CausalPredictor must then recover the masked slots' states from the observable slots—forcing cross-object causal reasoning.

$$\mathcal{L}_{\text{JEPA}} = \mathcal{L}_{\text{SIGReg}}\big(\text{Predictor}(\text{MaskedSlots}), \text{TargetEncoder}(\text{AllSlots})\big)$$

No external label identifies which slot corresponds to which source. The model discovers the decomposition because **only by tracking each source independently can it predict each source's future correctly.** This is object-centric physical modeling emerging from prediction pressure alone.

---

### 4. The Unimodal Physical Prior (UPP) Hypothesis

**Scenario A (alignment-first):** An audio encoder trained only to discriminate 51 sound classes, then aligned with a visual encoder via contrastive loss. The audio representation encodes *what* but not *how*—it knows "helicopter" but not whether the helicopter is approaching or receding. Alignment with vision cannot inject this knowledge; the visual encoder knows spatial motion, but there is no mechanism for it to *teach* the audio encoder—only to pull representations closer in a shared space.

**Scenario B (prediction-first, our approach):** The audio encoder first learns a temporal world model: volume increase ≈ decreasing distance, Doppler downshift ≈ approaching velocity, reverberation increase ≈ enclosed space, low-frequency attenuation ≈ occlusion. Only then does cross-modal alignment occur. At this point, alignment is confirmation, not instruction:

> Audio: "I detect an object moving toward me at approximately 15 m/s."
> Vision: "I confirm: it is a motorcycle, blue, at bearing 30°."

We formalize this as the **Unimodal Physical Prior (UPP) hypothesis:**

$$\text{AlignmentQuality}(M_A, M_V) \leq \min\big(\text{PhysicalUnderstanding}(M_A), \text{PhysicalUnderstanding}(M_V)\big)$$

Cross-modal alignment cannot compensate for deficits in unimodal physical modeling. The ceiling is set by the weaker modality. Training effort should be front-loaded: maximize each modality's independent grasp of physical dynamics before any cross-modal objective is introduced.

---

### 5. Empirical Instantiation: 51-Class Acoustic Event Classification

We instantiate this framework on a 51-class acoustic classification task:

**Data:** ESC-50 (2000 labeled clips, 50 environmental sound categories across 5-second duration) + Deep House music (177 tracks, 10% stratified subsample from 1,582) as a single "non-event" anchor class.

**Preprocessing:** 128-bin Mel spectrograms at 22,050 Hz, 2,048-point FFT, 512-sample hop. All spectrograms precomputed and cached to disk (3,776 `.pt` files), achieving ~110× throughput improvement over real-time decoding.

**Architecture:** AudioFeatureProjector (Conv2D → BatchNorm → GELU) → CAJEPA (5 Object Slots with per-slot differentiable projection, Object-Level Temporal Masker with full-history masking, Causal Transformer Predictor).

**Training Objective:**

$$\mathcal{L} = \mathcal{L}_{\text{SIGReg}}(\hat{z}_{\text{future}}, z_{\text{future}}) + \lambda \cdot \mathcal{L}_{\text{InfoNCE}}(z_{\text{clip}}, y)$$

where $\lambda = 0.3$ balances self-supervised temporal prediction and supervised contrastive classification.

**Evaluation:** 5-fold linear probe on frozen projector features every 5 epochs, measuring 51-class accuracy using ESC-50 fold 5 as held-out validation. Training completes in under one hour on a single GPU.

---

### 6. Design Principle and Core Equation

```text
Perception(x) = f_cochlea ∘ f_attention ∘ f_prediction(x)

where:
  f_cochlea    : Mel-scale filterbank → biological compression
  f_attention  : Learned selection of predictive features
  f_prediction : Forward-model generating future-state expectations

AlignmentQuality(M_A, M_V) ≤ min(PhysicalUnderstanding(M_A), PhysicalUnderstanding(M_V))
```

**Good auditory representation**
**= knowing what to preserve + knowing what to discard + knowing what comes next**
**(Mel/Cochlea)               (Encoder learning)           (JEPA prediction task)**

---

### 7. Conclusion

We have argued, formalized, and empirically instantiated the following principle:

> *Cross-modal alignment is a pull in representation space, not a transfer of physical knowledge. The ceiling of alignment quality is determined by the depth of unimodal pretraining, not by the sophistication of the alignment mechanism. Investment in single-modality physical modeling must precede investment in cross-modal fusion.*

For auditory intelligence specifically: the ear must first learn that volume rises when things approach, that pitch drops when sources recede, that reverberation thickens in enclosed spaces, and that multiple sources can be tracked independently. Only then does alignment with vision become semantic confirmation rather than shallow feature stitching.

A compelling empirical test of this principle comes from human audition itself: **congenitally blind individuals build full physical world models—distance, spatial layout, motion, causality—purely through hearing, independent of vision.** They navigate rooms, estimate crowd sizes, and operate computers via auditory interfaces. Analogously, unimodal audio encoders can learn complete physical regularities without visual supervision. This carries a direct engineering consequence: *audition-first architectures are inherently accessible.* Vision-first architectures, by contrast, implicitly require visual input as a prerequisite to understanding—a barrier for visually impaired users. In our framework, cross-modal alignment is neither necessary nor sufficient for sensory intelligence. It is confirmation, not instruction.

**The ear is not a microphone. It is a prediction engine wearing a filterbank.**

---

### References

1. LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. *OpenReview.*
2. Assran, M. et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. *CVPR.*
3. Bardes, A. et al. (2024). Revisiting Feature Prediction for Learning Visual Representations from Video. *arXiv:2404.08471.*
4. Locatello, F. et al. (2020). Object-Centric Learning with Slot Attention. *NeurIPS.*
5. Piczak, K. (2015). ESC: Dataset for Environmental Sound Classification. *ACM Multimedia.*
6. Terver, B. et al. (2026). EB-JEPA: A Lightweight Library for Energy-Based Joint Embedding Predictive Architectures. *ICLR Workshop on World Models.*
7. Nam, H. et al. (2026). Causal-JEPA: Learning World Models through Object-Level Latent Interventions. *arXiv:2602.11389.*
8. Maes, L. et al. (2026). LeWorldModel: End-to-End JEPA World Models from Pixels. *arXiv:2603.19312.*

---

*VORTEX FLAME Project — CAJEPA Audio Pipeline*
*Repository: github.com/maco1979/VORTEX_FLAME*
