---
name: ascend-operator-instrumentation
description: 为昇腾算子自动添加或更新打点代码，生成可编译可用的 TRACE_POINT/MoeTracing 插桩，并遵循函数级粒度、最多7层、智能合并等规则。用户提到算子打点、性能点位、MoeTracing 或 TRACE_POINT 时使用。
---

# 昇腾算子自动打点

## 目标

根据自然语言需求，为目标算子生成可落地的算子侧打点代码。

边界约束：
- 本 skill **负责** 算子代码插桩 + profiling 数据采集/解析工具链的完整闭环。
- 本 skill **不修改** 算子的业务逻辑（matmul、通信等功能代码），仅新增 profiling 相关代码。
- 本 skill **需要支持** 在仅有算子代码时，自动补齐打点所需工程脚本、编译接入、以及从 profiling tensor 到 Chrome Trace JSON 的完整处理链路。

**推荐执行顺序（与下方步骤编号对应）**：扫描与规划（1→2→3）→ 插桩（4）→ 静态校验（5）→ 部署工具链与编译接入（6）→ Profile 测试脚本分叉（7，可与 6 并行准备，但须在 pybind/算子已暴露 profiling 输出之后才有意义）。

## 输入

- 目标算子路径，例如 `src/.../op_kernel/<op>.h`。
- 打点风格：`MoeTracing(TRACE_POINT("label", "B/E"))` 或带上下文 `MoeTracing(TRACE_POINT("label", "B/E"), extraId, index)`。
- 约束条件：
  - 函数级粒度（见下方"打点密度与均匀性要求"）
  - 根节点名称固定为 `processing`
  - 最大深度为 7（实际按语义需要决定，不要人为卡在浅层）
  - 对深层或低价值调用链执行智能合并

## 必须执行的流程

1. **扫描目标代码**
   - 从入口文件出发，**递归跟随 `#include` 进入同算子目录下的所有头文件**，直到遍历完整个算子内部代码树。不能只看入口 `.h`，必须读取其直接或间接包含的所有实现文件。
   - 识别主流程阶段与函数边界；特别关注 **模板实例化调用链**：如果入口函数调用了模板类并最终执行 `operator()()`，该 `operator()` 同样属于主流程阶段边界，必须跟进到对应头文件。
   - 识别 **AIC / AIV 分核执行路径**：如果算子使用混合核（1C2V 等），AIC 分支和 AIV 分支各自是独立的主流程，需要分别打点。
   - 对于 1C2V 等模式，**必须检查 `operator()<AIV>()` 内部是否存在角色分工**（如 send core / recv core / compute core / share quant core）。不同 AIV 核可能通过 `aivIdx` 或 `GetSubBlockIdx()` 走完全不同的分支，每种角色的主要工作阶段都需要独立打点。
   - 尽量保留已存在且合法的点位。

2. **构建打点树**
   - L1 必须是 `processing`。
   - L2 至 L7 必须来源于当前算子真实语义（不要把 `dispatch/combine` 当作全局默认词）；合并规则见步骤 3，语义需要时用到 L6/L7 是正常的。
   - 对 AIC/AIV 分核执行路径，分别用 `<phase> aic` / `<phase> aiv` 作为 L2/L3 区分。
   - 对 expert group 循环、stage 循环等带索引的重复结构，打点时必须传递索引参数（见下方 MoeTracing 规格）。

3. **应用智能合并规则**
   - 超过 7 层的调用，折叠到最近的 L7 祖先节点。
   - 对无同步/无通信边界的薄封装函数与 helper 进行合并。
   - 对热点语义（`wait`、`sync`、`send`、`recv`、`copy`、`quant`、`dequant`）保留独立点位。

4. **插入代码**
   - 使用稳定命名的 `B/E` 成对点位。
   - 保证 begin/end 词法嵌套正确。
   - **"最内层循环"指 tile 级别的矩阵计算循环（如 blockMmad 内部的 K 轴迭代），不要在其中打点**。但 expert group 循环、stage 循环属于阶段边界，必须在循环体入口/出口打点。
   - 区分"阶段边界"与"tile 内层"——**同一个 gemm kernel 文件中既包含阶段边界也包含 tile 内层，不能因为文件名含 gemm/epilogue 就整体跳过**：
     - ✅ 需要打点：gemm kernel 文件中的 `operator()<AIC/AIV>()` 函数入口与出口（它们是 AIC/AIV 分核执行的阶段边界）、每组 expert 的循环入口、AIC↔AIV 同步等待、epilogue 计算、combine/dispatch 各子阶段。
     - ❌ 不要打点：同一文件中 BlockMmad / BlockEpilogue 内部的 tile 级 L0/L1 搬运循环、单个 DataCopy 调用。
     - 判断标准：如果一个函数是 **整个阶段的调度入口**（如 `operator()<AIC>()`），则打点；如果是 **单次 tile 计算的内层实现**（如 `blockMmad()` 内部的 K 轴迭代），则不打点。

5. **校验**
   - 对改动文件运行 `scripts/validate_trace_points.py`，检查点位命名与 B/E 配对。
   - 运行 `scripts/check_compile_safety.py <operator_dir>`，静态检查插桩是否会引入编译错误。此脚本检查：花括号平衡、预处理指令配对（`#if`/`#endif`）、MoeTracing 头文件可达性、TRACE_POINT 参数语法、变量作用域、profiling guard 闭合、kernel 参数与 op_host 注册的一致性。
   - 如果校验失败，修正问题后重新运行。两个脚本都通过后才能进入下一步。

6. **部署工具链并接入编译（必须执行，不可跳过）**
   - 此步骤不是可选的"缺省场景"，而是打点流程的必要组成部分。即使插桩代码已正确插入，如果工具链脚本未部署、预处理未接入编译，打点数据无法采集和解析。
   - **发现 build 目录**：在项目中搜索编译脚本（如 `compile*.sh`、`build*.sh`、`Makefile`、`CMakeLists.txt`），定位算子的 build 目录。常见位置如 `build/`、`scripts/` 等，不要假设目录名称。
   - **部署脚本**：运行 `bootstrap_trace_toolchain.py` 将 `trace_preprocessor.py`、`trace_utils.py`、`trace_save.py`、`trace_collector.py` 复制到 build 目录。
   - **接入编译**：运行 `patch_build_pipeline.py` 在编译脚本中注入预处理 hook。如果 patch 脚本的 anchor 与实际编译脚本不匹配（函数名大小写不同、路径变量不同等），需要手工在编译脚本的 `build.sh` 调用之前插入预处理命令，并用 `# TRACE_PREPROCESSOR_HOOK_START` / `# TRACE_PREPROCESSOR_HOOK_END` 标记包裹。
   - **校验部署**：运行 `verify_trace_scaffold.py` 确认脚本文件存在且编译 hook 已就位。
   - 不覆盖用户已有脚本；已存在时只做缺失补齐或可控更新。

7. **Profile 测试脚本分叉（推荐，不污染原 UT）**
   - **目的**：原始 `test_*.py` / `*_sample.py` 往往按**固定个数**解包主输出；插桩后在原有输出之外**多一个** `profiling_data`（总个数 = 原个数 + 1）。应在**不修改原文件**的前提下，拷贝一份专门用于采集与工具链联调的脚本。
   - **命名与位置**：
     - 若原文件为 `test/cam/ut/<op>/test_<op>.py`，在同目录新增 `test_<op>_profile.py`（在**原文件名 stem 后**加 `_profile`，保留扩展名）。
     - 若原文件为 `src/cam/examples/<op>_sample.py`，可新增 `<op>_sample_profile.py`，或按项目惯例放在 `test/` 下，但需与团队约定一致。
   - **拷贝后必改内容**：
     - 所有 `torch.ops.<lib>.<op>(...)` 的返回值解包在**原有主输出之后多接一个** profiling 张量，例如 fused_deep_moe 常见为：`..., profiling = torch.ops....`（主输出个数因算子而异，以 `op_host` 注册为准）。
     - 封装算子的 `nn.Module`（如 `FusionOp`）的 `_apply_ops` / `forward` 若向上传递元组，需**整条调用链** arity +1，避免中间层仍按旧个数解包。
     - **SmallOps / 对照路径**：若 baseline 小算子链不返回 profiling，可保持原元组长度不变；仅 fusion 路径多接 `profiling` 并在主流程对比时按索引对齐（参考下方示例）。
   - **与工具链对接（参考 `umdk_trace` 示例）**：部署后的 `build/.../trace_utils.py` 提供 `get_enable_moe_profiling()`（从头文件读 `ENABLE_MOE_PROFILING`）和 `save_profiling_data`。典型写法见 `umdk_trace/src/cam/examples/fused_deep_moe_sample.py`：
     - `FusionOp._apply_ops` 内对 `fused_deep_moe` 解包「原主输出 + profiling」并 `return (..., profiling)`。
     - 主函数/测试末尾：解包 fusion 路径结果时**比原 UT 多一个** profiling 变量名（如 fused_deep_moe：`fused_op_token_output, fused_op_share_output, fused_op_count_output, profiling = fused_op_output`），具体变量个数随算子主输出个数而变。
     - `if trace_utils.get_enable_moe_profiling(): profiling = profiling.cpu(); trace_utils.save_profiling_data(profiling, global_rank_id)`。若部署的 `save_profiling_data` 支持 `base_h_path=`，应对非默认目录下的 `<op>_base.h` 传入绝对路径，避免仍解析 fused_deep_moe 的固定相对路径。
   - **本仓库可参考实现**：`umdk/test/cam/ut/fused_deep_moe/test_fused_deep_moe_profile.py`（相对原 UT 使用独立类名如 `FusionOpWithProfile`、`test_base_test_profile`，并对 `import trace_utils` 做容错，无工具链时仍可跑数值对比）。
   - **`trace_utils` 导入**：在 profile 脚本开头将已部署工具链目录加入 `sys.path`（例如 `build/cam/comm_operator`），再 `import trace_utils`；或使用相对路径指向 skill 已部署副本，**不要**假设 cwd 固定。目录不存在或扩展未编译环境无 `torch` 时可用 `try/except ImportError` 将 `trace_utils` 置为 `None`，仅跳过落盘。
   - **环境说明**：`get_enable_moe_profiling()` 依赖能解析到目标算子 `_base.h` 中的 `ENABLE_MOE_PROFILING`；若默认相对路径指向别的算子或文件不存在，会静默为「不落盘」。需要落盘时传入 `base_h_path`（若 API 支持）、或修正 `trace_utils` 内查找逻辑、或在 profile 脚本中 `os.chdir` 到能使相对路径成立的工作目录（以不破坏原 UT 为前提）。
   - **pytest**：若原 UT 使用 `pytest` 与 `conftest`，profile 文件可单独用 `pytest test_<op>_profile.py` 运行，或在文件内用不同 `test_*` 函数名避免与原版重复收集（按项目习惯二选一）。

## MoeTracing 运行时规格

MoeTracing 不是简单的空宏。当项目的 base 头文件中缺少 MoeTracing 定义时，必须按以下规格在算子已有的 `_base.h` 文件中补齐（不要新建单独的头文件）。

### 宏定义

```cpp
#define ENABLE_MOE_PROFILING 1
#define PROF_SIZE_PER_CORE 2048
#define ENABLE_MOE_PROFILING_BARRIER true
```

### per-core profiling buffer 指针

每个核拥有独立的 profiling buffer，通过 block-local 指针访问：

```cpp
__BLOCK_LOCAL__ __inline__ int64_t* g_moeProfilePtr;

__aicore__ inline int64_t* GetMoeProfilePtr(uint32_t idx = 0)
{
    return &g_moeProfilePtr[idx];
}
```

如果算子存在 AIC/AIV 分核编译（`SPLIT_CORE_CUBE` / `SPLIT_CORE_VEC`），需要为每种核类型声明独立的指针变量（`g_moeProfilePtrCube` / `g_moeProfilePtrVec`），并在 `GetMoeProfilePtr()` 中根据编译宏选择。

### MoeTracing 函数实现

MoeTracing 是 **模板函数**，不是宏。模板参数 `sync` 控制是否在记录前插入 `PipeBarrier<PIPE_ALL>()`：

```cpp
template <bool sync = ENABLE_MOE_PROFILING_BARRIER>
__aicore__ inline void MoeTracingWithCycle(int64_t data, int64_t cycle)
{
#if ENABLE_MOE_PROFILING
    if constexpr (sync) {
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    int64_t *profileData = GetMoeProfilePtr();
    profileData[profileData[0]++] = data;
    profileData[PROF_SIZE_PER_CORE - profileData[0]] = cycle;
#endif
}
```

Buffer 布局：`profileData[0]` 是写入索引，**正向写 point_id 数据，反向写 cycle 时间戳**。

### 三种调用重载

```cpp
// 基础调用：记录 point_id + 当前 cycle
template <bool sync = ENABLE_MOE_PROFILING_BARRIER>
__aicore__ inline void MoeTracing(int64_t data)
{
    MoeTracingWithCycle<sync>(data, AscendC::GetSystemCycle());
}

// 带索引：将 index 编码到 data 高 32 位，用于区分不同 expert group / stage
template <bool sync = ENABLE_MOE_PROFILING_BARRIER>
__aicore__ inline void MoeTracing(int64_t data, uint32_t index)
{
    MoeTracing<sync>(data | (int64_t)(((uint64_t)index) << 32));
}

// 带 extraId + index：用于同时传递 stageId 和循环索引
template <bool sync = ENABLE_MOE_PROFILING_BARRIER>
__aicore__ inline void MoeTracing(int64_t data, uint32_t extraId, uint32_t index)
{
    MoeTracing<sync>(data, (extraId | (index << 8)));
}
```

### 调用示例

```cpp
// 基础打点
MoeTracing(TRACE_POINT("dispatch-gmm1 aic", "B"));

// 带 groupIdx（区分不同 expert）
MoeTracing(TRACE_POINT("dispatch-gmm1 moe-process", "B"), 0, groupIdx);

// 强制 barrier 后再记录（覆盖默认 sync 参数）
MoeTracing<true>(TRACE_POINT("gmm2-combine combine-barrier-all", "E"));
```

## 命名规则

- 通用根标签固定为 `processing`。
- 阶段标签必须从当前算子语义中提取。
- 标签采用 **空格分隔的层级路径**，前缀表示所属阶段，后缀表示具体子阶段。例如 `"dispatch-gmm1 aic"` 表示 dispatch-gmm1 阶段的 AIC 分支。
- 名称描述"做什么"，不要过度绑定实现细节。
- 在语义不变时，尽量保持命名稳定。

示例：
- `processing`
- `dispatch-gmm1`
- `dispatch-gmm1 aic`、`dispatch-gmm1 aiv`
- `dispatch-gmm1 moe-process`（带 groupIdx）
- `dispatch-gmm1 wait-moe-token`（带 groupIdx）
- `gmm2-combine block-epilogue waiting`（带 stageId）
- `gmm2-combine block-epilogue calc`（带 stageId）
- `gmm2-combine combine-send`、`gmm2-combine combine-recv`

## Profiling 数据搬运规格

打点数据写入 per-core 栈上 buffer 后，需要一条完整链路将其搬到 Host 侧。本 skill 要求在算子框架上**显式新增一个 profiling 输出 tensor**，而不是复用已有输入 tensor 的 GM 地址。

### 1. 算子框架层：新增 profiling 输出（在既有 output 之后多注册一个）

在 `op_host` 算子定义中新增一个 OPTIONAL output：

```cpp
// op_host/<op>.cpp — 算子注册
this->Output("profiling_data")
    .ParamType(OPTIONAL)
    .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
```

在 `op_host/<op>_infer.cpp` 中推导 shape：

**`totalCores` 必须等于 AIC 核数 + AIV 核数，不能只取其中一种。** kernel 侧 buffer 布局为：AIC 核写 `[0, aicNum)` 区间，AIV 核写 `[aicNum, aicNum + aivNum)` 区间。如果 `totalCores` 只取了 AIC 数量，AIV 核的写入会越界。

**禁止硬编码核数。** 不同硬件型号的核数不同（如 24C+48V=72、20C+40V=60），硬编码任何具体数字都会在其他型号上出错。正确做法按优先级：

1. **tiling 函数**（推荐）：通过 `platform_ascendc::PlatformAscendC(context.GetPlatformInfo())` 获取 `GetCoreNumAic()` 和 `GetCoreNumAiv()`，已有的 tiling 流程通常已包含此逻辑。
2. **kernel 入口**：通过 `AscendC::GetBlockNum()` 获取实际 AIC 核数，AIV 核数 = `GetBlockNum() * GetSubBlockNum()`（1C2V 下 SubBlockNum=2）。
3. **infer / pybind 侧**：`InferShapeContext` 没有平台查询 API，无法动态获取核数。应使用一个覆盖所有已知硬件的安全上界（如 `MAX_TOTAL_CORES = 128`），多分配的内存开销可忽略（128 × 2048 × 8B = 2MB）。

```cpp
// op_host/<op>_infer.cpp — 使用安全上界
constexpr uint32_t MAX_TOTAL_CORES = 128;
gert::Shape *profilingShape = context->GetOutputShape(OUTPUT_PROFILING_DATA);
profilingShape->SetDimNum(1);
profilingShape->SetDim(0, MAX_TOTAL_CORES * PROF_SIZE_PER_CORE);
context->SetOutputDataType(OUTPUT_PROFILING_DATA, ge::DT_INT64);
```

```cpp
// pybind — 同样使用安全上界；在原有 return 张量列表末尾追加 profilingData（以下为 fused_deep_moe 示例）
constexpr int64_t MAX_TOTAL_CORES = 128;
at::Tensor profilingData = at::zeros({MAX_TOTAL_CORES * PROF_SIZE_PER_CORE}, opts.dtype(at::kLong));
return {output, shareOutput, expertTokenNums, profilingData};
```

```cpp
// kernel 入口 — 使用 AscendC API 动态确定写入位置（已自动适配所有硬件）
if (g_coreType == AscendC::AIC) {
    coreGlobal = profGlobal[AscendC::GetBlockIdx() * PROF_SIZE_PER_CORE];
} else {
    coreGlobal = profGlobal[(AscendC::GetBlockNum() + AscendC::GetBlockIdx()) * PROF_SIZE_PER_CORE];
}
```

### 2. Kernel 入口（`.cpp`）：buffer 初始化 + 搬出

在 kernel 入口函数中，算子执行**前后**分别处理 profiling buffer：

**执行前**——在栈上分配 buffer、初始化写索引和起始时间戳、设置指针：

```cpp
#if ENABLE_MOE_PROFILING
    int64_t profData[PROF_SIZE_PER_CORE];
    profData[0] = 1;
    profData[PROF_SIZE_PER_CORE - 1] = AscendC::GetSystemCycle();
    SetMoeProfilePtr(&profData[0]);
#endif
```

**执行后**——将栈上 buffer 逐条写到 profiling output tensor 的 GM 地址：

```cpp
#if ENABLE_MOE_PROFILING
    AscendC::GlobalTensor<int64_t> profGlobal;
    profGlobal.SetGlobalBuffer((__gm__ int64_t *)(profiling_data));
    // AIC 核写前半段，AIV 核写后半段
    AscendC::GlobalTensor<int64_t> coreGlobal;
    if (g_coreType == AscendC::AIC) {
        coreGlobal = profGlobal[AscendC::GetBlockIdx() * PROF_SIZE_PER_CORE];
    } else {
        coreGlobal = profGlobal[(AscendC::GetBlockNum() + AscendC::GetBlockIdx()) * PROF_SIZE_PER_CORE];
    }
    for (unsigned i = 0; i < profData[0]; ++i) {
        coreGlobal(i) = profData[i];
        coreGlobal(PROF_SIZE_PER_CORE - i - 1) = profData[PROF_SIZE_PER_CORE - i - 1];
    }
    // DataCacheCleanAndInvalid 确保 host 可读
#endif
```

辅助函数 `SetMoeProfilePtr` 的定义放在 `.cpp` 入口文件中，根据分核编译宏选择正确的 block-local 指针：

```cpp
__aicore__ inline void SetMoeProfilePtr(int64_t *profilePtr)
{
#if __CCE_AICORE__ == 220 || defined(__DAV_C310__) || defined(__DAV_310R6__)
#ifdef SPLIT_CORE_CUBE
    g_moeProfilePtrCube = profilePtr;
#elif defined(SPLIT_CORE_VEC)
    g_moeProfilePtrVec = profilePtr;
#else
    g_moeProfilePtr = profilePtr;
#endif
#else
    g_moeProfilePtr = profilePtr;
#endif
}
```

### 3. Host 侧（Python）：读取 + 保存

`trace_utils.py` 中 `save_profiling_data` 需要：
- 从 `_base.h` 读取 `PROF_SIZE_PER_CORE` 与 `ENABLE_MOE_PROFILING`（若实现里提供 `base_h_path` 参数，**新算子应传入当前算子的 `<op>_base.h` 绝对路径**，避免工具链内写死的相对路径仍指向示例算子）。
- 将 profiling tensor reshape 为 `(total_cores, PROF_SIZE_PER_CORE)`；`get_core_num_list()` 等若仍为示例中的硬编码或环境变量，需与目标硬件/tiling 一致，否则分组索引会越界或切分错误。
- 按 AIC/AIV 核类型分组（考虑 1C2V 映射）
- 保存为 `rank{id}.pt` 供后续解析工具使用

```python
profiling = profiling_data.cpu()
trace_utils.save_profiling_data(profiling, rank_id, "profiling_data")  # 第三个参数为输出目录；若 API 支持可传 base_h_path=...
```

## 打点密度与均匀性要求

- **目标标签数**：每种核类型（AIC / AIV）各约 **15-20 个标签**。不要过少（<10 个看不到子阶段瓶颈）也不要过多（>30 个 buffer 压力大且 trace 可读性差）。
- **均匀性**：各主阶段（dispatch、gmm1、gmm2、combine 等）的标签数量应大致均衡。如果通信阶段有 6 个子标签而计算阶段只有 2 个，说明计算部分打点不足，需要深入 `operator()` 内部补充子阶段。
- **"函数级粒度"的正确理解**：指每个有独立语义的阶段函数（如 `SendCoreFunc`、`RecvCoreFunc`、`CompCoreFunc`、`UpdateAndCleanInfo`），不是仅限于调用链第一层入口的 `operator()`。`operator()` 内部如果有多个语义明确的子函数调用，每个都应该有独立的 B/E 点位。
- **二级拆分**：即使一个子函数已经有了 B/E 点位（如 `gmm1 aiv send`），如果它的内部存在语义可分离的子阶段（如 count-prep vs token-DMA、spin-wait vs data-copy），也应该在函数内部进一步添加子标签（如 `gmm1 aiv send-count` + `gmm1 aiv send-token`）。典型的可拆分模式包括：
  - **count/status 广播** vs **payload 数据搬运**（dispatch、recv）
  - **spin-wait/polling** vs **实际计算或搬运**（recv-count、group-wait）
  - **shared expert** vs **routed expert** 的独立执行路径
  - **metadata load**（index/scale DataCopyPad）vs **per-token reduce 循环**（combine local-copy）
- **AIV 角色分工**：对于 1C2V 等混合核模式，`operator()<AIV>()` 内部可能通过 `aivIdx`、`GetSubBlockIdx()` 或角色标志（`isSendCore`、`isRecvCore`、`isCompCore`）将不同 AIV 核分配到不同工作路径。每种角色的主要工作阶段都需要独立打点，让 trace 中能区分各类 AIV 核的时间分布。
- **多变体对齐**：如果同一算子有多个 kernel 变体（如 deep-fuse vs shallow-dispatch），所有变体的 AIV `operator()` 都应该有相似粒度的子阶段标签。不能一个变体有 8 个子标签而另一个只有入口/出口。
- **自检方法**：打点完成后，分别列出 AIC 和 AIV 能看到的标签清单。如果某个核类型的标签数 < 15，或者某个主阶段（如 gmm1、gmm2）内部只有入口/出口两个点位而没有子阶段，则需要补充。

## 容量与扰动约束

- 每核 profiling buffer 容量有限（`PROF_SIZE_PER_CORE`），禁止默认高密度铺点。
- 不要默认给每个小 helper 或最内层循环都加点。
- 优先保证可读性与稳定定位瓶颈能力，而不是追求全覆盖。

## 常见陷阱（快速自检）

- **因「gemm 文件名」跳过打点**：`grouped_matmul_*.h` 内 `operator()<AIC/AIV>()` 仍是阶段边界，必须与 tile 内层循环区分（见步骤 4）。
- **infer/pybind 硬编码核数**：用安全上界或动态逻辑；与 kernel 侧 per-core 写入区间一致。
- **Python 解包 arity**：插桩后 fusion 路径比原先多一个 profiling 张量；原 UT 不解包改时直接拷贝为 `test_<op>_profile.py`（步骤 7）。
- **`trace_utils` 静默不落盘**：`_base.h` 路径不对或 `ENABLE_MOE_PROFILING` 为 0；优先检查 `base_h_path` 与宏。
- **工具链未部署仍以为能出 trace**：步骤 6 未完成则没有预处理后的 `point_map.json` 与可复现的 point_id。

## trace.json 生成流程

打点数据的完整处理链路（从设备到可视化）分 4 步：

### Step 1: 预处理（编译前）

`trace_preprocessor.py` 扫描源码，将 `TRACE_POINT("label", "B/E")` 替换为唯一整数 point_id，生成 `point_map.json`：

```bash
python <skill_root>/scripts/trace_preprocessor.py <operator_src_dir> <output_dir> --modify
```

输出 `point_map.json` 格式：
```json
{
  "points": {
    "1": {"label": "processing", "event_type": "B", "file": "...", "line": 415, "event_id": 1},
    "2": {"label": "dispatch-gmm1 aic", "event_type": "B", ...}
  }
}
```

### Step 2: 运行算子采集 profiling tensor

算子执行后，Host 侧获取 profiling output tensor（通常为**最后一个** output，即比插桩前多出来的那一个），调用 `trace_utils.save_profiling_data` 拆分保存：

```python
import trace_utils
profiling = profiling_data.cpu()
trace_utils.save_profiling_data(profiling, rank_id, "profiling_data")
```

也可离线保存：
```bash
python <skill_root>/scripts/trace_save.py raw_profiling.pt --rank 0 --output profiling_data
```

### Step 3: 生成 Chrome Trace JSON

`trace_collector.py` 读取所有 `rank*.pt` + `point_map.json`，解析 64 位组合 ID（低 32 位 point_id + 高 32 位 extra_id），配对 B/E 事件，生成 Chrome Trace 格式：

```bash
python <skill_root>/scripts/trace_collector.py profiling_data point_map.json -o chrome_trace.json
```

支持参数：
- `--clock-divisor 50.0`：时钟频率 MHz（cycle → us 换算）
- `--extra-mode seq`：extra_id 解析模式（`seq`=高 24 位序号+低 8 位 extra，`legacy`=整体使用）
- `--depth 0`：区间深度过滤（0=全部，1=仅叶子，2=叶子+父层）

### Step 4: 可视化

在 Chrome 浏览器打开 `chrome://tracing`，加载生成的 `chrome_trace.json`。
每个 rank 对应一个 process，每个核（AIC/AIV × core_id）对应一个 thread。

## 固定脚本

路径规范：
- 所有示例命令使用**相对路径**。
- 不依赖 Cursor 专属绝对路径，便于在其他环境复用。

**与「必须执行的流程」步骤对应（检索用）**：

| 步骤 | 脚本或产物 |
|------|----------------|
| 1–4 辅助（可选） | `generate_instrumentation_plan.py`、`instrument_operator.py` — 规划/草稿插桩，不能替代人工审查 |
| **5 校验** | `validate_trace_points.py`、`check_compile_safety.py` |
| **6 工具链 + 编译接入** | `bootstrap_trace_toolchain.py` → `patch_build_pipeline.py` 或手工 hook → `verify_trace_scaffold.py`；一键 `apply_trace_scaffold.sh`。编译前在源码目录跑 `trace_preprocessor.py ... --modify`，生成 `point_map.json` |
| 运行后解析 | `trace_save.py`（离线 `.pt`）、`trace_collector.py`（→ `chrome_trace.json`）— 见上文「trace.json 生成流程」 |
| **7 Profile UT** | 无独立脚本；分叉出的 `test_<op>_profile.py` 通过 `sys.path` 引用已部署目录中的 `trace_utils.py` |

- 生成打点草案（函数树 + 合并决策）：
  - `python <skill_root>/scripts/generate_instrumentation_plan.py --root <operator_dir> --entry <entry_function>`
- 根据函数边界自动写入打点代码：
  - `python <skill_root>/scripts/instrument_operator.py --target <operator_src_file_or_dir> --root-label processing`
- 校验点位命名与 B/E 配对：
  - `python <skill_root>/scripts/validate_trace_points.py <file_or_dir>`
- 静态编译安全检查（花括号平衡、预处理配对、头文件可达、参数一致性等）：
  - `python <skill_root>/scripts/check_compile_safety.py <operator_dir>`
  - 加 `--strict` 将 warnings 也视为错误
- 预处理（编译前替换 TRACE_POINT 为整数 ID）：
  - `python <skill_root>/scripts/trace_preprocessor.py <operator_src_dir> <output_dir> --modify`
- 保存 profiling tensor（离线）：
  - `python <skill_root>/scripts/trace_save.py <raw_pt_file> --rank <rank_id> --output <profiling_data_dir>`
- 生成 Chrome Trace JSON：
  - `python <skill_root>/scripts/trace_collector.py <profiling_data_dir> <point_map.json> -o chrome_trace.json`
- 部署工具链脚本到 build 目录：
  - `python <skill_root>/scripts/bootstrap_trace_toolchain.py --build-dir <build_module_dir>`
- 对编译脚本打补丁并接入预处理（幂等）：
  - `python <skill_root>/scripts/patch_build_pipeline.py --compile-script <compile_script_path> --preprocessor-cmd "<cmd>"`
- 校验工具链与编译接入是否就绪：
  - `python <skill_root>/scripts/verify_trace_scaffold.py --build-dir <build_module_dir> --compile-script <compile_script_path>`
- 一键执行"部署工具链 + 编译接入 + 校验"：
  - `bash <skill_root>/scripts/apply_trace_scaffold.sh <skill_root> <build_module_dir> <compile_script_path>`

## 输出约定

完成后必须给出：
- 哪些文件被插桩修改。
- 最终点位层级结构（L1 为 `processing`，子层随语义展开，通常不超过 L7；合并后可在说明中标注折叠关系）。
- `validate_trace_points.py` 的校验结果。
- `check_compile_safety.py` 的校验结果。
- 工具链部署结果（哪些脚本被复制到 build 目录）。
- 若已按步骤 7 分叉测试脚本：新文件路径（如 `test_<op>_profile.py`）及相对原 UT 的改动要点（比原先多一个 profiling 返回值、`trace_utils` 保存逻辑）。
- trace.json 生成命令示例。
