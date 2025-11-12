## Landmark LLM scaling papers
Megatron-LM
Introduces tensor parallelism and efficient model parallelism techniques for training large language models.

Megatron-Turing NLG 530B
Describes the training of a 530B parameter model using a combination of DeepSpeed and Megatron-LM frameworks.

PaLM
Introduces Google's Pathways Language Model, demonstrating strong performance across hundreds of language tasks and reasoning capabilities.

Gemini
Presents Google's multimodal model architecture capable of processing text, images, audio, and video inputs.

Llama 3
Introduces the Llama 3 herd of models.

DeepSeek-V3
DeepSeek's report on the architecture and training of the DeepSeek-V3 model.

## Training frameworks
Nanotron
Our framework for training large language models, featuring various parallelism strategies.

Megatron-LM
NVIDIA's framework for training large language models, featuring various parallelism strategies.

DeepSpeed
Microsoft's deep learning optimization library, featuring ZeRO optimization stages and various parallelism strategies.

FairScale
A PyTorch extension library for large-scale training, offering various parallelism and optimization techniques.

Colossal-AI
An integrated large-scale model training system with various optimization techniques.

torchtitan
A PyTorch native library for large model training.

GPT-NeoX
EleutherAI's framework for training large language models, used to train GPT-NeoX-20B.

LitGPT
Lightning AI's implementation of 20+ state-of-the-art open source LLMs, with a focus on reproducibility.

OpenDiLoCo
An open source framework for training language models across compute clusters with DiLoCo.

torchgpipe
A GPipe implementation in PyTorch.

OSLO
The Open Source for Large-scale Optimization framework for large-scale modeling.

## Debugging
Speed profiling
Official PyTorch tutorial on using the profiler to analyze model performance and bottlenecks.

Memory profiling
Comprehensive guide to understanding and optimizing GPU memory usage in PyTorch.

Memory profiling walkthrough on a simple example
Guide to visualizing and understanding GPU memory in PyTorch.

TensorBoard profiler tutorial
Guide to using TensorBoard's profiling tools for PyTorch models.

## Distribution techniques
Data parallelism
Comprehensive explanation of data parallel training in deep learning.

ZeRO
Introduces the Zero Redundancy Optimizer for training large models with memory optimization.

FSDP
Fully Sharded Data Parallel training implementation in PyTorch.

Tensor and sequence parallelism + selective recomputation
Advanced techniques for efficient large-scale model training combining different parallelism strategies.

Pipeline parallelism
NVIDIA's guide to implementing pipeline parallelism for large model training.

Breadth-first pipeline parallelism
Includes broad discussions of PP schedules.

Ring all-reduce
Detailed explanation of the ring all-reduce algorithm used in distributed training.

Ring Flash Attention
Implementation of the Ring Attention mechanism combined with FlashAttention for efficient training.

Ring Attention tutorial
Tutorial explaining the concepts and implementation of Ring Attention.

ZeRO and 3D
DeepSpeed's guide to understanding the trade-offs between ZeRO and 3D parallelism strategies.

Mixed precision training
Introduces mixed precision training techniques for deep learning models.

Visualizing 6D mesh parallelism
Explains the collective communication involved in a 6D parallel mesh.

## Hardware
Fire-Flyer, a 10,000 PCI chip cluster
DeepSeek's report on designing a cluster with 10k PCI GPUs.

Meta's 24k H100 clusters
Meta's detailed overview of their massive AI infrastructure built with NVIDIA H100 GPUs.

SemiAnalysis's 100k H100 cluster
Analysis of large-scale H100 GPU clusters and their implications for AI infrastructure.

Modal GPU glossary
CUDA docs for humans.

## Others
Stas Bekman's handbook
Comprehensive handbook covering various aspects of training LLMs.

BLOOM training chronicles
Detailed documentation of the BLOOM model training process and challenges.

OPT logbook
Meta's detailed logbook documenting the training process of the OPT-175B model.

Harm's law for training smol models longer
Investigation of the relationship between model size and training overhead.

Harm's blog on long contexts
Investigation of long context training in terms of data and training cost.

GPU Mode
A GPU reading group and community.

EleutherAI YouTube channel
ML scalability & performance reading group.

Google JAX scaling book
How to scale your model.

@fvsmassa & @TimDarcet FSDP
Standalone ~500 LoC FSDP implementation

thonking.ai
Some of Horace He's blog posts.

Aleksa's ELI5: FlashAttention
Easy explanation of FlashAttention.

TunibAI's 3D parallelism tutorials
Large-scale language modeling tutorials with PyTorch. 