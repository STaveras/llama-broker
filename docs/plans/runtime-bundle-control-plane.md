# Runtime Bundle Control Plane Implementation Plan

**Goal:** Add deterministic, isolated project inspection and reviewed CPU bundle build commands without changing production runtime selection or inference transitions.

**Architecture:** `runtime_control.py` owns validated source metadata, subprocess argv execution, reviewed CMake CPU recipe, immutable bundle publication, and manifest verification. `brokerctl runtime project` and `brokerctl runtime bundle` expose JSON-first machine-readable operations alongside existing `runtime list`.

**Scope:** Initial recipe is llama.cpp-derived CMake with CUDA explicitly OFF; network is opt-in and build commands are never shell-evaluated. Dependency closure and broader recipe support remain follow-up work.
