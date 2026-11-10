"""
Backdoor Antigen: Immunization-inspired backdoor purification for LLMs.

Paper: "Purifying Generative LLMs from Backdoors Without Prior Knowledge or Clean Reference"
       (ICLR 2026)

Core modules:
    - data_builder: Construct variant datasets with different key-behavior pairs
    - residual: Compute differential deltas (Eq. 1)
    - scoring: Magnitude-and-consistency scoring (Eq. 2)
    - suppression: Neuron suppression for LoRA and full-model settings
"""
