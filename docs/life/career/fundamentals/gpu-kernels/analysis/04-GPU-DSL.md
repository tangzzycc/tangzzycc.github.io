---
title: "GPU DSL：FlyDSL 与 CuTe"
---

# GPU DSL：FlyDSL 与 CuTe

## 1. FlyDSL 是什么？它与 CuTe 有什么共同点？

1. **Layout 是核心。** 用 shape/stride 和组合、分块、分区等操作表达“哪个坐标对应哪个地址、哪个线程持有哪些值”。Layout view 的变化本身不代表实际搬运数据。
2. **Copy/MMA atom 对应底层操作。** Copy atom 表达搬运方式，MMA atom 表达矩阵指令及 fragment 分布；TiledCopy/TiledMma 将它们扩展到更大 tile，让搬运和计算的数据布局匹配。
3. **抽象相似，硬件约束不同。** AMD 的 wave、LDS 和 WMMA/MFMA 与 NVIDIA 的 warp、shared memory 和矩阵指令不同，迁移仍需核对布局、同步和流水线。
4. **编译路径。** FlyDSL 用 Python 构造并编译 GPU 程序，同一 specialization 可复用编译产物；不能将它与 Python eager 执行等同，也不能把所有 CuTe 实现都说成同一条 MLIR 编译链。

扩展阅读：[FlyDSL 编程模型与编译流程](../../../../../gpu-kernel/flydsl/01-FlyDSL编程模型与编译流程.md)、[Layout 与 Tile 编程](../../../../../gpu-kernel/flydsl/02-Layout代数与Tile编程.md)。
