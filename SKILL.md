---
name: ascend-operator-instrumentation
description: 为昇腾算子自动添加或更新打点代码，生成可编译可用的 TRACE_POINT/MoeTracing 插桩，并遵循函数级粒度、最多5层、智能合并等规则。用户提到算子打点、性能点位、MoeTracing 或 TRACE_POINT 时使用。
---

# 昇腾算子自动打点

## 目标

根据自然语言需求，为目标算子生成可落地的算子侧打点代码。

边界约束：
- 本 skill **只负责** 算子代码插桩。
- 本 skill **不允许** 修改 `trace.json` 的解析/收集逻辑。
- 本 skill **需要支持** 在仅有算子代码时，自动补齐打点所需工程脚本与编译接入。

## 输入

- 目标算子路径，例如 `src/.../op_kernel/<op>.h`。
- 打点风格：`MoeTracing(TRACE_POINT("label", "B/E"), ...)`。
- 约束条件：
  - 默认函数级粒度
  - 根节点名称固定为 `processing`
  - 最大深度为 5
  - 对深层或低价值调用链执行智能合并

## 必须执行的流程

1. **扫描目标代码**
   - 识别主流程阶段与函数边界。
   - 尽量保留已存在且合法的点位。

2. **构建打点树**
   - L1 必须是 `processing`。
   - L2-L5 必须来源于当前算子真实语义（不要把 `dispatch/combine` 当作全局默认词）。

3. **应用智能合并规则**
   - 超过 5 层的调用，折叠到最近的 L5 祖先节点。
   - 对无同步/无通信边界的薄封装函数与 helper 进行合并。
   - 对热点语义（`wait`、`sync`、`send`、`recv`、`copy`、`quant`、`dequant`）保留独立点位。

4. **插入代码**
   - 使用稳定命名的 `B/E` 成对点位。
   - 保证 begin/end 词法嵌套正确。
   - 优先在阶段边界、函数边界打点，避免默认深入最内层循环。

5. **校验**
   - 对改动文件运行 `scripts/validate_trace_points.py`。
   - 如果校验失败，修正命名与配对后重新运行。

6. **补齐工程能力（缺省场景）**
   - 当项目不存在打点工具链时，自动生成最小可用脚本集合（预处理、保存、转换）。
   - 对已有编译工程自动执行幂等补丁：把预处理脚本接入到算子编译流程。
   - 不覆盖用户已有脚本；已存在时只做缺失补齐或可控更新。

## 命名规则

- 通用根标签固定为 `processing`。
- 阶段标签必须从当前算子语义中提取。
- 使用简短短语，建议 `kebab-case`。
- 名称描述“做什么”，不要过度绑定实现细节。
- 在语义不变时，尽量保持命名稳定。

示例：
- `processing`
- `dispatch-gmm1`
- `dispatch init`
- `dispatch process`
- `gmm2-combine block-epilogue calc`

## 容量与扰动约束

- 每核 profiling buffer 容量有限（`PROF_SIZE_PER_CORE`），禁止默认高密度铺点。
- 不要默认给每个小 helper 或最内层循环都加点。
- 优先保证可读性与稳定定位瓶颈能力，而不是追求全覆盖。

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
- 生成打点工具链文件（用于“只有算子代码”的工程）：
  - `python <skill_root>/scripts/bootstrap_trace_toolchain.py --build-dir <build_module_dir>`
- 对编译脚本打补丁并接入预处理（幂等）：
  - `python <skill_root>/scripts/patch_build_pipeline.py --compile-script <compile_script_path> --preprocessor-cmd "<cmd>"`
- 校验工具链与编译接入是否就绪：
  - `python <skill_root>/scripts/verify_trace_scaffold.py --build-dir <build_module_dir> --compile-script <compile_script_path>`
- 一键执行“生成工具链 + 编译接入 + 校验”：
  - `bash <skill_root>/scripts/apply_trace_scaffold.sh <skill_root> <build_module_dir> <compile_script_path>`

## 输出约定

完成后必须给出：
- 哪些文件被插桩修改。
- 最终点位层级结构（L1-L5）。
- `validate_trace_points.py` 的校验结果。
- 工具链生成/编译接入结果（新增了哪些脚本、修改了哪些构建文件）。
