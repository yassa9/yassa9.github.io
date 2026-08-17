---
layout: default
title: projects
permalink: /projects/
---

![pc](/images/pc.jpg){: .writeup-img style="width:36%;" }

# SIDE PROJECTS

Side opensource projects I do in low level programming in `C++` and `CUDA`.

<div class="project-sep"></div>

<hr class="project-sep">

<div class="project-row" markdown="1">
<img src="/images/projects/dvltcu.jpg" alt="dvlt.cu output">
<div class="project-text" markdown="1">

## [dvlt.cu](https://github.com/yassa9/dvlt.cu)

Suckless, single binary, zero-dependency CUDA/C++ inference engine for NVIDIA's DVLT. Reconstructs 3D scenes from a handful of images (depth + rays + camera pose => point cloud), no python, no torch, no framework.

</div>
</div>

<hr class="project-sep">

<div class="project-row reverse" markdown="1">
<img src="/images/projects/frokenizer-bench.jpg" alt="frokenizer benchmark chart">
<div class="project-text" markdown="1">

## [frokenizer](https://github.com/yassa9/frokenizer)

Zero allocation, zero dependency, header only C++ BPE tokenizer for Qwen, using ahead-of-time DFA compilation to eliminate regex backtracking and heap overhead, reaching GBs/sec tokenization throughput.

</div>
</div>

<hr class="project-sep">

<div class="project-row" markdown="1">
<img src="/images/projects/q600-demo.jpg" alt="qwen600.cu demo">
<div class="project-text" markdown="1">

## [qwen600.cu](https://github.com/yassa9/qwen600)

+500 stars on github. Static, single batch inference engine for `QWEN3-0.6B` written in pure CUDA C/C++, no python dependencies. Faster than llama.cpp by ~8.5%.

</div>
</div>
