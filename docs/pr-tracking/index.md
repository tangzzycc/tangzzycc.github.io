---
title: "开源 PR 追踪"
hide:
  - navigation
  - toc
---

<div class="pr-tracker" markdown="1">

# 开源 PR 追踪

<div class="pr-tracker-meta">
  <a href="https://github.com/tangzzycc">@tangzzycc</a>
  <span>更新于 2026-10-06</span>
  <a class="pr-tracker-github" href="https://github.com/tangzzycc">GitHub <span aria-hidden="true">↗</span></a>
</div>

## vLLM <span class="pr-section-count">8</span> { #vllm }

<div class="pr-card-grid pr-project--vllm" markdown="0">
<article class="pr-card pr-card--open" aria-label="vllm-project/vllm #60049">
  <div class="pr-card-top">
    <span class="pr-card-identity">
      <a class="pr-card-repo" href="https://github.com/vllm-project/vllm" title="vllm-project/vllm">vLLM</a>
      <a class="pr-card-number" href="https://github.com/vllm-project/vllm/pull/60049">#60049</a>
      <span class="pr-card-tag">Bugfix</span><span class="pr-card-tag">MRV2</span>
    </span>
    <span class="pr-card-status"><span class="pr-status-dot" aria-hidden="true"></span>Open</span>
  </div>
  <h3 class="pr-card-title"><a href="https://github.com/vllm-project/vllm/pull/60049" title="[Bugfix][MRV2] Preserve dummy tokens across PIECEWISE padding">Preserve dummy tokens across PIECEWISE padding</a></h3>
  <p class="pr-card-summary">区分有效 token 与 PIECEWISE 填充，避免自适应投机解码启动时超出编译范围。</p>
  <div class="pr-card-footer">
    <span class="pr-card-date">提交于 <time datetime="2026-10-05">2026-10-05</time></span>
    <a class="pr-card-note" href="../vllm/%E5%BC%80%E5%8F%91%E5%AE%9E%E8%B7%B5/MRV2%E7%9A%84PIECEWISE%E5%A1%AB%E5%85%85%E4%B8%8E%E6%8A%95%E6%9C%BA%E8%A7%A3%E7%A0%81%E5%90%AF%E5%8A%A8Profiling/">学习笔记 <span aria-hidden="true">→</span></a>
  </div>
</article>
<article class="pr-card pr-card--open" aria-label="vllm-project/vllm #60040">
  <div class="pr-card-top">
    <span class="pr-card-identity">
      <a class="pr-card-repo" href="https://github.com/vllm-project/vllm" title="vllm-project/vllm">vLLM</a>
      <a class="pr-card-number" href="https://github.com/vllm-project/vllm/pull/60040">#60040</a>
      <span class="pr-card-tag">ROCm</span><span class="pr-card-tag">Attention</span>
    </span>
    <span class="pr-card-status"><span class="pr-status-dot" aria-hidden="true"></span>Open</span>
  </div>
  <h3 class="pr-card-title"><a href="https://github.com/vllm-project/vllm/pull/60040" title="[ROCm][Attention] Support multimodal Prefix-LM in ROCM_ATTN">Support multimodal Prefix-LM in ROCM_ATTN</a></h3>
  <p class="pr-card-summary">为 Gemma3 等模型补齐图像 token 双向注意力，串起范围元数据、分页 KV 与图执行。</p>
  <div class="pr-card-footer">
    <span class="pr-card-date">提交于 <time datetime="2026-10-05">2026-10-05</time></span>
    <a class="pr-card-note" href="../vllm/%E5%BC%80%E5%8F%91%E5%AE%9E%E8%B7%B5/%E5%A4%9A%E6%A8%A1%E6%80%81Prefix-LM%E5%9C%A8ROCm-Attention%E4%B8%AD%E7%9A%84%E5%AE%9E%E7%8E%B0/">学习笔记 <span aria-hidden="true">→</span></a>
  </div>
</article>
<article class="pr-card pr-card--open" aria-label="vllm-project/vllm #59600">
  <div class="pr-card-top">
    <span class="pr-card-identity">
      <a class="pr-card-repo" href="https://github.com/vllm-project/vllm" title="vllm-project/vllm">vLLM</a>
      <a class="pr-card-number" href="https://github.com/vllm-project/vllm/pull/59600">#59600</a>
      <span class="pr-card-tag">Bugfix</span><span class="pr-card-tag">Spec Decode</span>
    </span>
    <span class="pr-card-status"><span class="pr-status-dot" aria-hidden="true"></span>Open</span>
  </div>
  <h3 class="pr-card-title"><a href="https://github.com/vllm-project/vllm/pull/59600" title="[Bugfix][Spec Decode] Restore ShortConv drafter state in MRV2">Restore ShortConv drafter state in MRV2</a></h3>
  <p class="pr-card-summary">在投机轮次间恢复 ShortConv 历史，并修正 Chunked Prefill 的下一步输入。</p>
  <div class="pr-card-footer">
    <span class="pr-card-date">提交于 <time datetime="2026-10-01">2026-10-01</time></span>
    <a class="pr-card-note" href="../vllm/%E5%BC%80%E5%8F%91%E5%AE%9E%E8%B7%B5/MRV2%E6%8A%95%E6%9C%BA%E8%A7%A3%E7%A0%81%E4%B8%AD%E7%9A%84ShortConv%E5%8E%86%E5%8F%B2%E6%81%A2%E5%A4%8D/">学习笔记 <span aria-hidden="true">→</span></a>
  </div>
</article>
<article class="pr-card pr-card--merged" aria-label="vllm-project/vllm #59550">
  <div class="pr-card-top">
    <span class="pr-card-identity">
      <a class="pr-card-repo" href="https://github.com/vllm-project/vllm" title="vllm-project/vllm">vLLM</a>
      <a class="pr-card-number" href="https://github.com/vllm-project/vllm/pull/59550">#59550</a>
      <span class="pr-card-tag">ROCm</span><span class="pr-card-tag">Bugfix</span>
    </span>
    <span class="pr-card-status"><span class="pr-status-dot" aria-hidden="true"></span>Merged</span>
  </div>
  <h3 class="pr-card-title"><a href="https://github.com/vllm-project/vllm/pull/59550" title="[ROCm][Bugfix] Fix ROCM_ATTN sliding-window boundary">Fix ROCM_ATTN sliding-window boundary</a></h3>
  <p class="pr-card-summary">将窗口左侧跨度换算为完整窗口长度，修复 ROCm Attention 漏读最早有效 token。</p>
  <div class="pr-card-footer">
    <span class="pr-card-date">合并于 <time datetime="2026-10-04">2026-10-04</time></span>
    <a class="pr-card-note" href="../vllm/%E5%BC%80%E5%8F%91%E5%AE%9E%E8%B7%B5/ROCm-Attention%E6%BB%91%E5%8A%A8%E7%AA%97%E5%8F%A3%E7%9A%84%E5%8F%82%E6%95%B0%E4%B8%8E%E8%BE%B9%E7%95%8C/">学习笔记 <span aria-hidden="true">→</span></a>
  </div>
</article>
<article class="pr-card pr-card--open" aria-label="vllm-project/vllm #58513">
  <div class="pr-card-top">
    <span class="pr-card-identity">
      <a class="pr-card-repo" href="https://github.com/vllm-project/vllm" title="vllm-project/vllm">vLLM</a>
      <a class="pr-card-number" href="https://github.com/vllm-project/vllm/pull/58513">#58513</a>
      <span class="pr-card-tag">Bugfix</span><span class="pr-card-tag">ROCm</span>
    </span>
    <span class="pr-card-status"><span class="pr-status-dot" aria-hidden="true"></span>Open</span>
  </div>
  <h3 class="pr-card-title"><a href="https://github.com/vllm-project/vllm/pull/58513" title="[Bugfix][ROCm] Fix channelwise GPTQ MoE fallback">Fix channelwise GPTQ MoE fallback</a></h3>
  <p class="pr-card-summary">修正按通道 GPTQ MoE 的权重分配和 scale 索引，让 Triton fallback 正确处理整通道量化。</p>
  <div class="pr-card-footer">
    <span class="pr-card-date">提交于 <time datetime="2026-09-24">2026-09-24</time></span>
    <a class="pr-card-note" href="../vllm/%E5%BC%80%E5%8F%91%E5%AE%9E%E8%B7%B5/Channelwise-GPTQ-MoE%E7%9A%84%E5%8A%A0%E8%BD%BD%E4%B8%8ETriton%E6%89%A7%E8%A1%8C/">学习笔记 <span aria-hidden="true">→</span></a>
  </div>
</article>
<article class="pr-card pr-card--merged" aria-label="vllm-project/vllm #53283">
  <div class="pr-card-top">
    <span class="pr-card-identity">
      <a class="pr-card-repo" href="https://github.com/vllm-project/vllm" title="vllm-project/vllm">vLLM</a>
      <a class="pr-card-number" href="https://github.com/vllm-project/vllm/pull/53283">#53283</a>
      <span class="pr-card-tag">ROCm</span><span class="pr-card-tag">Perf</span>
    </span>
    <span class="pr-card-status"><span class="pr-status-dot" aria-hidden="true"></span>Merged</span>
  </div>
  <h3 class="pr-card-title"><a href="https://github.com/vllm-project/vllm/pull/53283" title="[ROCm][Perf] Use wvSplitK for single-output GEMMs">Use wvSplitK for single-output GEMMs</a></h3>
  <p class="pr-card-summary">让单输出 GEMM 进入已有 wvSplitK 路径，加速 Qwen Shared Expert 的门控计算。</p>
  <div class="pr-card-footer">
    <span class="pr-card-date">合并于 <time datetime="2026-09-23">2026-09-23</time></span>
    <a class="pr-card-note" href="../vllm/%E5%BC%80%E5%8F%91%E5%AE%9E%E8%B7%B5/%E5%8D%95%E8%BE%93%E5%87%BAGEMM%E4%B8%8EShared-Expert%E9%97%A8%E6%8E%A7%E7%9A%84ROCm%E4%BC%98%E5%8C%96/">学习笔记 <span aria-hidden="true">→</span></a>
  </div>
</article>
<article class="pr-card pr-card--open" aria-label="vllm-project/vllm #52637">
  <div class="pr-card-top">
    <span class="pr-card-identity">
      <a class="pr-card-repo" href="https://github.com/vllm-project/vllm" title="vllm-project/vllm">vLLM</a>
      <a class="pr-card-number" href="https://github.com/vllm-project/vllm/pull/52637">#52637</a>
      <span class="pr-card-tag">Quantization</span><span class="pr-card-tag">ROCm</span>
    </span>
    <span class="pr-card-status"><span class="pr-status-dot" aria-hidden="true"></span>Open</span>
  </div>
  <h3 class="pr-card-title"><a href="https://github.com/vllm-project/vllm/pull/52637" title="[Quantization][ROCm] Fix symmetric GPTQ loading and optimize RDNA INT4 repacking">Fix symmetric GPTQ loading and optimize RDNA INT4 repacking</a></h3>
  <p class="pr-card-summary">清理对称 GPTQ 的冗余零点，复用布局重排，减少 RDNA INT4 权重加载开销。</p>
  <div class="pr-card-footer">
    <span class="pr-card-date">提交于 <time datetime="2026-08-18">2026-08-18</time></span>
    <a class="pr-card-note" href="../vllm/%E5%BC%80%E5%8F%91%E5%AE%9E%E8%B7%B5/GPTQ%E6%9D%83%E9%87%8D%E5%8A%A0%E8%BD%BD%E4%B8%8ERDNA-INT4%E5%B8%83%E5%B1%80%E9%87%8D%E6%8E%92/">学习笔记 <span aria-hidden="true">→</span></a>
  </div>
</article>
<article class="pr-card pr-card--open" aria-label="vllm-project/vllm #52619">
  <div class="pr-card-top">
    <span class="pr-card-identity">
      <a class="pr-card-repo" href="https://github.com/vllm-project/vllm" title="vllm-project/vllm">vLLM</a>
      <a class="pr-card-number" href="https://github.com/vllm-project/vllm/pull/52619">#52619</a>
      <span class="pr-card-tag">ROCm</span><span class="pr-card-tag">Perf</span>
    </span>
    <span class="pr-card-status"><span class="pr-status-dot" aria-hidden="true"></span>Open</span>
  </div>
  <h3 class="pr-card-title"><a href="https://github.com/vllm-project/vllm/pull/52619" title="[ROCm][Perf] Enable medium-skinny dispatch in RDNAHybridW4A16LinearKernel">Enable medium-skinny dispatch in RDNAHybridW4A16LinearKernel</a></h3>
  <p class="pr-card-summary">对齐 Python 与 HIP 的分发边界，让 medium-skinny 形状使用已有 INT4 kernel。</p>
  <div class="pr-card-footer">
    <span class="pr-card-date">提交于 <time datetime="2026-08-17">2026-08-17</time></span>
    <a class="pr-card-note" href="../vllm/%E5%BC%80%E5%8F%91%E5%AE%9E%E8%B7%B5/RDNA-W4A16%E7%9A%84%E5%BD%A2%E7%8A%B6%E5%88%86%E5%8F%91%E4%B8%8ELDS%E8%BE%B9%E7%95%8C/">学习笔记 <span aria-hidden="true">→</span></a>
  </div>
</article>
</div>

## FlyDSL <span class="pr-section-count">3</span> { #flydsl }

<div class="pr-card-grid pr-project--flydsl" markdown="0">
<article class="pr-card pr-card--open" aria-label="ROCm/FlyDSL #1113">
  <div class="pr-card-top">
    <span class="pr-card-identity">
      <a class="pr-card-repo" href="https://github.com/ROCm/FlyDSL" title="ROCm/FlyDSL">FlyDSL</a>
      <a class="pr-card-number" href="https://github.com/ROCm/FlyDSL/pull/1113">#1113</a>
      <span class="pr-card-tag">Bugfix</span><span class="pr-card-tag">Kernel</span>
    </span>
    <span class="pr-card-status"><span class="pr-status-dot" aria-hidden="true"></span>Open</span>
  </div>
  <h3 class="pr-card-title"><a href="https://github.com/ROCm/FlyDSL/pull/1113" title="[Bugfix][Kernel] Fix fused RoPE for head dimension 96">Fix fused RoPE for head dimension 96</a></h3>
  <p class="pr-card-summary">调整向量宽度与 NeoX lane 配对，使融合 RoPE 支持 head dimension 96。</p>
  <div class="pr-card-footer">
    <span class="pr-card-date">提交于 <time datetime="2026-09-09">2026-09-09</time></span>
    <a class="pr-card-note" href="../gpu-kernel/flydsl/FlyDSL-1113-Fused-RoPE-Head-Dim-96/">学习笔记 <span aria-hidden="true">→</span></a>
  </div>
</article>
<article class="pr-card pr-card--open" aria-label="ROCm/FlyDSL #1108">
  <div class="pr-card-top">
    <span class="pr-card-identity">
      <a class="pr-card-repo" href="https://github.com/ROCm/FlyDSL" title="ROCm/FlyDSL">FlyDSL</a>
      <a class="pr-card-number" href="https://github.com/ROCm/FlyDSL/pull/1108">#1108</a>
      <span class="pr-card-tag">Bugfix</span><span class="pr-card-tag">Benchmark</span>
    </span>
    <span class="pr-card-status"><span class="pr-status-dot" aria-hidden="true"></span>Open</span>
  </div>
  <h3 class="pr-card-title"><a href="https://github.com/ROCm/FlyDSL/pull/1108" title="[Bugfix][Benchmark] Fix compiler API usage in Softmax and RMSNorm benchmarks">Fix compiler API usage in Softmax and RMSNorm benchmarks</a></h3>
  <p class="pr-card-summary">将 Softmax、RMSNorm 的旧编译调用迁至 compiler API，恢复缺失的计时结果。</p>
  <div class="pr-card-footer">
    <span class="pr-card-date">提交于 <time datetime="2026-09-09">2026-09-09</time></span>
    <span class="pr-card-arrow" aria-hidden="true">↗</span>
  </div>
</article>
<article class="pr-card pr-card--open" aria-label="ROCm/FlyDSL #1057">
  <div class="pr-card-top">
    <span class="pr-card-identity">
      <a class="pr-card-repo" href="https://github.com/ROCm/FlyDSL" title="ROCm/FlyDSL">FlyDSL</a>
      <a class="pr-card-number" href="https://github.com/ROCm/FlyDSL/pull/1057">#1057</a>
      <span class="pr-card-tag">Kernel</span><span class="pr-card-tag">Perf</span>
    </span>
    <span class="pr-card-status"><span class="pr-status-dot" aria-hidden="true"></span>Open</span>
  </div>
  <h3 class="pr-card-title"><a href="https://github.com/ROCm/FlyDSL/pull/1057" title="[Kernel][Perf] Add gfx1151 tile selection for RDNA3 GEMM">Add gfx1151 tile selection for RDNA3 GEMM</a></h3>
  <p class="pr-card-summary">为 gfx1151 选择合适的 GEMM Tile，并统一设备架构、JIT target 与缓存标识。</p>
  <div class="pr-card-footer">
    <span class="pr-card-date">提交于 <time datetime="2026-08-22">2026-08-22</time></span>
    <a class="pr-card-note" href="../gpu-kernel/flydsl/FlyDSL-1057-RDNA3-GEMM-Tile-Policy/">学习笔记 <span aria-hidden="true">→</span></a>
  </div>
</article>
</div>

## FlashRT <span class="pr-section-count">2</span> { #flashrt }

<div class="pr-card-grid pr-project--flashrt" markdown="0">
<article class="pr-card pr-card--merged" aria-label="flashrt-project/FlashRT #215">
  <div class="pr-card-top">
    <span class="pr-card-identity">
      <a class="pr-card-repo" href="https://github.com/flashrt-project/FlashRT" title="flashrt-project/FlashRT">FlashRT</a>
      <a class="pr-card-number" href="https://github.com/flashrt-project/FlashRT/pull/215">#215</a>
      
    </span>
    <span class="pr-card-status"><span class="pr-status-dot" aria-hidden="true"></span>Merged</span>
  </div>
  <h3 class="pr-card-title"><a href="https://github.com/flashrt-project/FlashRT/pull/215" title="Add Temporal K/V reuse for RDNA 3.5 Pi0.5">Add Temporal K/V reuse for RDNA 3.5 Pi0.5</a></h3>
  <p class="pr-card-summary">连续帧复用视觉语言 Prefix K/V，中间帧只执行 Decoder；修改提示词后刷新缓存。</p>
  <div class="pr-card-footer">
    <span class="pr-card-date">合并于 <time datetime="2026-09-26">2026-09-26</time></span>
    <a class="pr-card-note" href="../gpu-kernel/flashrt-rdna35/06-Temporal-KV%E8%B7%A8%E5%B8%A7%E5%A4%8D%E7%94%A8/">学习笔记 <span aria-hidden="true">→</span></a>
  </div>
</article>
<article class="pr-card pr-card--merged" aria-label="flashrt-project/FlashRT #213">
  <div class="pr-card-top">
    <span class="pr-card-identity">
      <a class="pr-card-repo" href="https://github.com/flashrt-project/FlashRT" title="flashrt-project/FlashRT">FlashRT</a>
      <a class="pr-card-number" href="https://github.com/flashrt-project/FlashRT/pull/213">#213</a>
      
    </span>
    <span class="pr-card-status"><span class="pr-status-dot" aria-hidden="true"></span>Merged</span>
  </div>
  <h3 class="pr-card-title"><a href="https://github.com/flashrt-project/FlashRT/pull/213" title="Add AMD RDNA 3.5 BF16 backend for Pi0.5">Add AMD RDNA 3.5 BF16 backend for Pi0.5</a></h3>
  <p class="pr-card-summary">为 Pi0.5 增加 RDNA 3.5 BF16 后端，接通共享 Pipeline、HIP 算子与 GEMM。</p>
  <div class="pr-card-footer">
    <span class="pr-card-date">合并于 <time datetime="2026-09-24">2026-09-24</time></span>
    <a class="pr-card-note" href="../gpu-kernel/flashrt-rdna35/">学习笔记 <span aria-hidden="true">→</span></a>
  </div>
</article>
</div>

## AITER <span class="pr-section-count">1</span> { #aiter }

<div class="pr-card-grid pr-project--aiter" markdown="0">
<article class="pr-card pr-card--open" aria-label="ROCm/aiter #4814">
  <div class="pr-card-top">
    <span class="pr-card-identity">
      <a class="pr-card-repo" href="https://github.com/ROCm/aiter" title="ROCm/aiter">AITER</a>
      <a class="pr-card-number" href="https://github.com/ROCm/aiter/pull/4814">#4814</a>
      <span class="pr-card-tag">Triton</span><span class="pr-card-tag">gfx1151</span>
    </span>
    <span class="pr-card-status"><span class="pr-status-dot" aria-hidden="true"></span>Open</span>
  </div>
  <h3 class="pr-card-title"><a href="https://github.com/ROCm/aiter/pull/4814" title="[Triton][gfx1151] Enable GEMM-A16W16">Enable GEMM-A16W16</a></h3>
  <p class="pr-card-summary">补齐 gfx1151 的 A16W16 默认配置，按 M 范围选择 Tile，启用已有 Triton GEMM。</p>
  <div class="pr-card-footer">
    <span class="pr-card-date">提交于 <time datetime="2026-08-18">2026-08-18</time></span>
    <span class="pr-card-arrow" aria-hidden="true">↗</span>
  </div>
</article>
</div>

## vLLM GGUF Plugin <span class="pr-section-count">1</span> { #vllm-gguf }

<div class="pr-card-grid pr-project--vllm-gguf" markdown="0">
<article class="pr-card pr-card--open" aria-label="vllm-project/vllm-gguf-plugin #140">
  <div class="pr-card-top">
    <span class="pr-card-identity">
      <a class="pr-card-repo" href="https://github.com/vllm-project/vllm-gguf-plugin" title="vllm-project/vllm-gguf-plugin">vLLM GGUF Plugin</a>
      <a class="pr-card-number" href="https://github.com/vllm-project/vllm-gguf-plugin/pull/140">#140</a>
      <span class="pr-card-tag">Model</span>
    </span>
    <span class="pr-card-status"><span class="pr-status-dot" aria-hidden="true"></span>Open</span>
  </div>
  <h3 class="pr-card-title"><a href="https://github.com/vllm-project/vllm-gguf-plugin/pull/140" title="[Model] Add Qwen-VL GGUF support">Add Qwen-VL GGUF support</a></h3>
  <p class="pr-card-summary">支持 Qwen2／2.5／3-VL 的 GGUF 权重加载，处理视觉、投影层与融合 QKV 布局。</p>
  <div class="pr-card-footer">
    <span class="pr-card-date">提交于 <time datetime="2026-09-26">2026-09-26</time></span>
    <span class="pr-card-arrow" aria-hidden="true">↗</span>
  </div>
</article>
</div>

</div>
