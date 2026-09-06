# vLLM CogAgent-18B support

## CogAgent-18B模型结构

从 vLLM 顶层类看，CogAgent-18B模型结构如下：

```python
CogAgentForCausalLM
│
├── 1. vision: EVA2CLIPModel
│      224×224 图片
│      → 258 个 4096 维视觉 embedding
│      → 替换 Decoder 中的 image placeholder
│
├── 2. cross_vision: CrossVisionModel
│      1120×1120 图片
│      → 6400 个 1024 维 source feature
│      → 被 32 层 Cross-Attention 读取
│
├── 3. model: CogAgentModel
│      token embedding
│      → CogAgentDecoderLayer × 32
│      → final RMSNorm
│
└── 4. lm_head
       4096 维最后位置 hidden state
       → 32000 维 logits
       → sampling / greedy next token

# vLLM code
self.model = CogAgentModel(...)
self.vision = EVA2CLIPModel(...)
self.cross_vision = CrossVisionModel(...)
self.lm_head = ParallelLMHead(vocab_size=32000, hidden_size=4096)
```

## 一次 Prefill 的顶层计算图

下面是 T=279（Prompt = 21 + Picture = 258 ） 示例从输入到 logits 的完整路径：

```python
messages + PIL image
        │
        ▼
CogAgentProcessor
        ├─ Decoder IDs [279]
        ├─ pixel_values       [1,3,224,224]
        └─ cross_pixel_values [1,3,1120,1120]
        │
        ├─────────────────────────────────────────────────────────┐
        ▼                                                         ▼
EVA2CLIPModel                                              CrossVisionModel
[1,3,224,224]                                              [1,3,1120,1120]
        │                                                         │
        ▼                                                         ▼
image_embeds [1,258,4096]                              cross_embeds [1,6400,1024]
        │                                                         │
        └──────────────┬──────────────────────────────────────────┘
                       ▼
MM Encoder Cache Entry [1,6658,4096]
                       │
                       ├─ [:258,:]      → image_embeds [1,258,4096]
                       └─ [258:,:1024]  → cross_embeds [1,6400,1024]
                       │
                       ▼
文本 Embedding [279,4096]
        + 替换 258 个 image placeholder
                       │
                       ▼
Decoder hidden [279,4096]
                       │
                       ▼
┌──────────────── CogAgentDecoderLayer × 32 ────────────────┐
│  Routed Self-Attention                                    │
│  Cross-Attention，读取 [6400,1024] source                  │  
│  Routed SwiGLU MLP                                        │
└───────────────────────────────────────────────────────────┘
                       │
                       ▼
Final RMSNorm [279,4096]
                       │
                       ▼
选择最后一个采样位置 [1,4096]
                       │
                       ▼
LM Head [1,32000] → next token
```

### CogAgentProcessor作用

> `CogAgentProcessor` 会在模型初始化后由 `CogAgentProcessingInfo.get_hf_processor()` 按模型配置动态创建，并在收到带图片的请求时经 `Renderer._process_multimodal()` 进入 `CogAgentMultiModalProcessor.apply()`，再由 `BaseMultiModalProcessor` 通过 `InputProcessingContext.call_hf_processor()` 调用其 `__call__()`，完成文本编码、BOS 补齐、图片转 RGB、分别缩放为 `224×224` 和 `1120×1120` 并生成 `pixel_values`/`cross_pixel_values`，之后 vLLM 再手动把 BOS 替换为图像占位 token，连同处理后的视觉张量、placeholder 和 encoder/decoder prompt 一起交给 CogAgent 模型推理。

Processor 不只负责 resize 图片，还在建立模型**计算图需要的三类长度**

```python
image
├─ resize/normalize → pixel_values       [1,3,224,224]
└─ resize/normalize → cross_pixel_values [1,3,1120,1120]
```

**Decoder Prompt 替换**：Processor 找到开头的 BOS，并替换为：\[BOS\] → \[BOS\] + 258 × \[image_token_id\]，因此原始 Prompt = 21 token 已经包含 BOS 时：处理后长度 = 21 - 1 + (1 + 258) = 279

**Encoder prompt 和三个长度**

| 长度 | 数值 | 真正用途 |
| --- | --- | --- |
| Decoder sequence length | 279 | Self-Attention 与 Decoder MLP 的 token 数 |
| Cross-Attention source length | 6400 | 每层 Cross-Attention 的 K/V 长度 |
| MM encoder output budget | 6658 | `258+6400`，两条视觉输出的统一缓存容器长度 |

### EVA2CLIPModel：224×224 小视觉塔的算子计算图

小视觉塔的目标不是产生 Cross-Attention K/V，而是产生可以直接放进 Decoder token 流的 4096 维 embedding、

顶层 Shape： [EVA2CLIPModel类实现](https://github.com/tangzzycc/vllm/blob/gfx11/cogagent-18b-support/vllm/model_executor/models/cogagent_vision_encoder.py#L736)

```python
Input image                         [1,3,224,224]
  │ Conv2d(k=14,s=14), out=1792
  ▼
Patch map                           [1,1792,16,16]
  │ flatten spatial + transpose
  ▼
Patch sequence                      [1,256,1792]
  │ prepend CLS + absolute position embedding
  ▼
Vision sequence                     [1,257,1792]
  │ EVATransformerLayer × 63
  ▼
Encoded sequence                    [1,257,1792]
  │ drop CLS
  ▼
Patch feature                       [1,256,1792]
  │ EVAGLU projection
  ▼
Decoder-width patch feature         [1,256,4096]
  │ prepend BOI + append EOI
  ▼
image_embeds                        [1,258,4096]
```