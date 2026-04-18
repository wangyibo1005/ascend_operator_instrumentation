---
name: ascend-operator-instrumentation
description: 为昇腾算子自动添加或更新打点代码，生成可编译可用的 TRACE_POINT/MoeTracing 插桩，并遵循函数级粒度、最多5层、智能合并等规则。用户提到算子打点、性能点位、MoeTracing 或 TRACE_POINT 时使用。
---

# 昇腾算子自动打点

## 目标

根据自然语言需求，为目标算子生成可落地的算子侧打点代码。

边界约束：
- 本 skill **负责** 算子代码插桩 + profiling 数据采集/解析工具链的完整闭环。
- 本 skill **不修改** 算子的业务逻辑（matmul、通信等功能代码），仅新增 profiling 相关代码。
- 本 skill **需要支持** 在仅有算子代码时，自动补齐打点所需工程脚本、编译接入、以及从 profiling tensor 到 Chrome Trace JSON 的完整处理链路。

## 输入

- 目标算子路径，例如 `src/.../op_kernel/<op>.h`。
- 打点风格：`MoeTracing(TRACE_POINT("label", "B/E"))` 或带上下文 `MoeTracing(TRACE_POINT("label", "B/E"), extraId, index)`。
- 约束条件：
  - 默认函数级粒度
  - 根节点名称固定为 `processing`
  - 最大深度为 5
  - 对深层或低价值调用链执行智能合并

## 必须执行的流程

1. **扫描目标代码**
   - 从入口文件出发，**递归跟随 `#include` 进入同算子目录下的所有头文件**，直到遍历完整个算子内部代码树。不能只看入口 `.h`，必须读取其直接或间接包含的所有实现文件。
   - 识别主流程阶段与函数边界；特别关注 **模板实例化调用链**：如果入口函数调用了模板类并最终执行 `operator()()`，该 `operator()` 同样属于主流程阶段边界，必须跟进到对应头文件。
   - 识别 **AIC / AIV 分核执行路径**：如果算子使用混合核（1C2V 等），AIC 分支和 AIV 分支各自是独立的主流程，需要分别打点。
   - 尽量保留已存在且合法的点位。

2. **构建打点树**
   - L1 必须是 `processing`。
   - L2-L5 必须来源于当前算子真实语义（不要把 `dispatch/combine` 当作全局默认词）。
   - 对 AIC/AIV 分核执行路径，分别用 `<phase> aic` / `<phase> aiv` 作为 L2/L3 区分。
   - 对 expert group 循环、stage 循环等带索引的重复结构，打点时必须传递索引参数（见下方 MoeTracing 规格）。

3. **应用智能合并规则**
   - 超过 5 层的调用，折叠到最近的 L5 祖先节点。
   - 对无同步/无通信边界的薄封装函数与 helper 进行合并。
   - 对热点语义（`wait`、`sync`、`send`、`recv`、`copy`、`quant`、`dequant`）保留独立点位。

4. **插入代码**
   - 使用稳定命名的 `B/E` 成对点位。
   - 保证 begin/end 词法嵌套正确。
   - **"最内层循环"指 tile 级别的矩阵计算循环（如 blockMmad 内部的 K 轴迭代），不要在其中打点**。但 expert group 循环、stage 循环属于阶段边界，必须在循环体入口/出口打点。
   - 区分"阶段边界"与"tile 内层"：
     - ✅ 需要打点：`operator()<AIC/AIV>()` 函数入口、每组 expert 的循环入口、AIC↔AIV 同步等待、epilogue 计算、combine/dispatch 各子阶段。
     - ❌ 不要打点：矩阵乘法 tile 内部的 L0/L1 搬运循环、单个 DataCopy 调用。

5. **校验**
   - 对改动文件运行 `scripts/validate_trace_points.py`。
   - 如果校验失败，修正命名与配对后重新运行。

6. **补齐工程能力（缺省场景）**
   - 当项目不存在打点工具链时，自动生成最小可用脚本集合（预处理、保存、转换）。
   - 对已有编译工程自动执行幂等补丁：把预处理脚本接入到算子编译流程。
   - 不覆盖用户已有脚本；已存在时只做缺失补齐或可控更新。

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

### 1. 算子框架层：新增第 4 个输出

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

```cpp
// shape = (total_cores * PROF_SIZE_PER_CORE,)
// total_cores = aicNum + aivNum（从 tiling 或 attr 获取）
constexpr uint32_t OUTPUT_PROFILING_DATA = 3;
gert::Shape *profilingShape = context->GetOutputShape(OUTPUT_PROFILING_DATA);
profilingShape->SetDimNum(1);
profilingShape->SetDim(0, totalCores * PROF_SIZE_PER_CORE);
context->SetOutputDataType(OUTPUT_PROFILING_DATA, ge::DT_INT64);
```

在 pybind 层分配 tensor 并返回：

```cpp
int64_t totalCores = aicNum + aivNum;
at::Tensor profilingData = at::zeros({totalCores * PROF_SIZE_PER_CORE}, opts.dtype(at::kLong));
// 传入 EXEC_NPU_CMD 并添加到返回值
return {output, shareOutput, expertTokenNums, profilingData};
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
- 从 `_base.h` 读取 `PROF_SIZE_PER_CORE` 和核数配置
- 将 profiling tensor reshape 为 `(total_cores, PROF_SIZE_PER_CORE)`
- 按 AIC/AIV 核类型分组（考虑 1C2V 映射）
- 保存为 `rank{id}.pt` 供后续解析工具使用

```python
profiling = profiling_data.cpu()
trace_utils.save_profiling_data(profiling, rank_id)
```

## 容量与扰动约束

- 每核 profiling buffer 容量有限（`PROF_SIZE_PER_CORE`），禁止默认高密度铺点。
- 不要默认给每个小 helper 或最内层循环都加点。
- 优先保证可读性与稳定定位瓶颈能力，而不是追求全覆盖。

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

算子执行后，Host 侧获取 profiling output tensor（第 4 个输出），调用 `trace_utils.save_profiling_data` 拆分保存：

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

- 生成打点草案（函数树 + 合并决策）：
  - `python <skill_root>/scripts/generate_instrumentation_plan.py --root <operator_dir> --entry <entry_function>`
- 根据函数边界自动写入打点代码：
  - `python <skill_root>/scripts/instrument_operator.py --target <operator_src_file_or_dir> --root-label processing`
- 校验点位命名与 B/E 配对：
  - `python <skill_root>/scripts/validate_trace_points.py <file_or_dir>`
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
- 最终点位层级结构（L1-L5）。
- `validate_trace_points.py` 的校验结果。
- 工具链部署结果（哪些脚本被复制到 build 目录）。
- trace.json 生成命令示例。
