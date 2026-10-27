# warmup 后学习率怎样降：min 两分支与 step 单位

lrate=d_model^-0.5*min(s^-0.5,s*w^-1.5)，w=4000 training steps。 这篇围绕一个新阅读问题，将D1–D4逐项核对。相同论文可以提出不同问题；本次调用与旧表格读数任务独立，不复用旧回答。

## 问题、方法与实测

完整问题见[task.txt](task.txt)，运行前标准见[criteria.md](criteria.md)，冻结哈希见[protocol-lock.json](protocol-lock.json)。只给所选论文片段，逐行编号，要求D1–D4 JSON；服务无历史无工具，先dry-run逐字核对message后真实调用一次。

模型：chenkun/gpt-5-mini；completion id：chatcmpl-EdW3OGGLh0QZvzKi1aechJoKpd9Sv；开始时间UTC：2026-10-27T07:38:32.461137+00:00；总耗时48.663秒；prompt/completion/total tokens：622/6580/7202。费用null，服务未返回逐次货币扣费。响应中的账户余额脱敏，模型content逐字保留；未伪造费用或实验。

机械验收：ranges_valid，5段合法范围，错误=[]。语义判定见每题核查；不将合法范围当作支持性证明。

输入[request.txt](request.txt)、预检[preflight.json](preflight.json)、响应[response.json](response.json)、原答[output.json](output.json)、元数据[metadata.json](metadata.json)、检查[validation.json](validation.json)均公开。

## 四道子题：原文、原答与作者核查

### D1：完整写公式(3)，解释 d_model、step_num、warmup_steps，4000 的单位是step还是epoch？给出Adam三个超参。

**结论**：lrate=d_model^-0.5*min(s^-0.5,s*w^-1.5)，w=4000 training steps。

**论文原文**：物理页7，P7:7–13。保留前后句如下：

```text
5.3 Optimizer
We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ϵ = 10 −9. We varied the learning
rate over the course of training, according to the formula:
lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5)) (3)
This corresponds to increasing the learning rate linearly for the first warmup_steps training steps,
and decreasing it thereafter proportionally to the inverse square root of the step number. We used
warmup_steps = 4000.
```

**模型原答（原样）**：

```text
公式(3)（逐字据文献）:
  lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
解释（仅依据给定证据）:
  - 公式中明确出现的符号有 d_model、step_num、warmup_steps；文献在公式之后说明“increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number.”（见证据），因此：
    • warmup_steps 表示预热阶段的步数（steps），文中给出 warmup_steps = 4000，故 4000 的单位是 training steps（步），不是 epoch（文献明示为 training steps）。
    • step_num 即训练步骤编号（step number），在公式中作为步数的变量（文献中将调度按 step 数变化）。
    • d_model 在该摘录中以符号出现于公式，但文献片段未在此处对 d_model 做定义（给定证据中没有对 d_model 的文字定义），因此关于 d_model 的意义（例如是否为模型隐藏维度）在本证据中未被明确定义 — 这一点属未在给定证据中定义的项。
  - Adam 的三项超参数（文献明示）:
    • β1 = 0.9
    • β2 = 0.98
    • ϵ = 10^(-9)
限制：d_model 的具体语义和 step_num 的起始编号（是否从 0 或 1 开始）在给定证据中未明确说明；此处只陈述文献中明示的内容。


模型 limitations字段：d_model 在给定证据片段中仅以符号出现但未被定义；step 编号的起点（是否允许 step=0）在本摘录未说明。其他对符号的常识性解释未被该证据显式给出，因此未在答案中断言。
```

**作者推导**：Adam β1=0.9、β2=0.98、epsilon=10^-9。s是优化步号，不是epoch。作者补充d_model是表示/模型宽度；该定义在本次给定P7片段没有明确文字，因此模型正确报告缺证据。d_model=512是题设教学代入，不声称来自P7。

**逐题核查**：原答对证据不足处理诚实；作者不会把事后常识定义伪装成原始输入。


### D2：解出两分支交点，说明哪支在warmup前/后被min选中；为什么step=0不能直接代负幂？

**结论**：交点s=w；预热取线性分支，之后取逆平方根。

**论文原文**：物理页7，P7:10–13。保留前后句如下：

```text
lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5)) (3)
This corresponds to increasing the learning rate linearly for the first warmup_steps training steps,
and decreasing it thereafter proportionally to the inverse square root of the step number. We used
warmup_steps = 4000.
```

**模型原答（原样）**：

```text
求两分支交点（推导）:
  由公式 lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))，两分支相等时满足
    step_num^(-0.5) = step_num * warmup_steps^(-1.5).
  推导：
    将等式两边乘以 step_num^{0.5} 得 1 = step_num^{1.5} * warmup_steps^{-1.5}，
    即 step_num^{1.5} = warmup_steps^{1.5}，
    取 2/3 次方（或直接开 1.5 次方反运算）得 step_num = warmup_steps.
  结论：交点恰好在 step_num = warmup_steps（即文中称的 warmup 步数处）。
哪支被 min 选中：
  - 在 warmup 之前（step_num < warmup_steps），min 选中第二项 step_num * warmup_steps^{-1.5}，因此学习率按 step_num 线性增加（文中明确："increasing the learning rate linearly for the first warmup_steps training steps"）。
  - 在 warmup 之后（step_num > warmup_steps），min 选中第一项 step_num^{-0.5}，因此学习率随步数按 1/sqrt(step_num) 减小（文中明确："decreasing it thereafter proportionally to the inverse square root of the step number"）。
为什么 step = 0 不能直接代入负幂（数学/定义层面说明）:
  - 分支中含有 step_num^{-0.5}，当 step_num = 0 时该项为 0 的负幂（除以零）而数学上未定义；因此从严格数学角度不能把 0 代入负幂项。文献没有在给定片段中说明 step 编号是否从 0 还是从 1 开始；因此实际实现通常会避免将 step=0 代入该表达式（例如从 step=1 开始或对 0 做特殊处理），但这类实现细节在给定证据中未说明。
限制：以上求交点与分支归属完全基于给定公式与文中对“线性增加/按逆平方根减少”的文字说明；关于 step 编号的实际起点和对 step=0 的实现处理，给定证据未提供具体实现细节或初始约定。 

模型 limitations字段：公式中包含 step_num^{-0.5}，step=0 导致数学不定（除以零），而文献片段未说明 step 编号的起点或对 step=0 的具体实现处理。交点推导为数学推导（推导），但实现细节在给定证据外。
```

**作者推导**：令s^-1/2=s*w^-3/2，得到s^(3/2)=w^(3/2)，正数域s=w。s<w时线性分支小。step=0的负幂未定义，代码显式要求s>0；若实现把初始lr设0，应是特殊约定。

**逐题核查**：原答解方程与分支方向正确。


### D3：d_model=512、warmup=4000时，计算step=1000/4000/16000的学习率与相对峰值比例。固定step时换d_model=1024会怎样？

**结论**：s=1000/4000/16000的lr约0.000174693/0.000698771/0.000349386，峰值比例0.25/1/0.5。

**论文原文**：物理页7，P7:10–13。保留前后句如下：

```text
lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5)) (3)
This corresponds to increasing the learning rate linearly for the first warmup_steps training steps,
and decreasing it thereafter proportionally to the inverse square root of the step number. We used
warmup_steps = 4000.
```

**模型原答（原样）**：

```text
数值计算（d_model = 512，warmup_steps = 4000；仅依据公式与给定 warmup 值；下列计算为推导/算术运算）：
  常量项：d_model^{-0.5} = 1/√512 ≈ 0.044194173824159216。
  warmup_steps^{-1.5} = 4000^{-1.5} ≈ 3.952847075210308e-6。
  1) step = 1000 (处于预热期，因为 1000 < 4000)：
     内部 min 的第二项为 1000 * warmup^{-1.5} ≈ 0.003952847075210308，故
     lrate ≈ 0.044194173824159216 * 0.003952847075210308 ≈ 0.000174663256 (约 1.7466e-4)。
     相对峰值（峰值在 step = warmup = 4000）：比例 = 1000/4000 = 0.25（即 25%）。
  2) step = 4000（处于交点 / 峰值点）：
     两项相等，s^{-0.5} = 4000^{-0.5} ≈ 0.015811388300841232，
     lrate ≈ 0.044194173824159216 * 0.015811388300841232 ≈ 0.000698653024 (约 6.9865e-4)。
     相对峰值比例 = 1（即为基准峰值）。
  3) step = 16000（在预热之后）：
     使用 s^{-0.5}，s^{-0.5} = 16000^{-0.5} ≈ 0.007905694150420616，
     lrate ≈ 0.044194173824159216 * 0.007905694150420616 ≈ 0.000349326512 (约 3.4933e-4)。
     相对峰值比例 = 0.000349326512 / 0.000698653024 = 0.5（即 50%），这与逆平方根衰减下比例 sqrt(4000/16000) = 1/2 一致。
  关于固定 step 时将 d_model 改为 1024 的影响：
    - 因为公式中 d_model 出现在乘法因子 d_model^{-0.5}，将 d_model 从 512 换为 1024 会把学习率乘以比率 (1024^{-0.5})/(512^{-0.5}) = 1/√2 ≈ 0.70710678。
    - 因此在相同 step（例如 1000、4000、16000）下，所有学习率值都会变为前述值的约 0.7071 倍。例如在 step=4000 时，原来约 0.000698653 -> 约 0.0004938。
限制：上述数值完全由给定公式与 warmup_steps = 4000 推导与计算得到；d_model 的解释与数值运算未超出该公式。实际训练中精确实现（如 step 的起点、数值精度或任何额外缩放）可能影响最终数值，但这些实现细节在给定证据中未给出。 

模型 limitations字段：所有数值均由论文给出的公式和 warmup_steps = 4000 计算得出；论文片段中未给出 d_model 的文字定义（仅出现在公式中）以及 step 编号起点等实现细节，故数值假定能直接代入公式并以常规数学含义计算。
```

**作者推导**：标准库复算为准，见offline-output.txt。固定s和w时，宽度512改1024，lr乘sqrt(512/1024)=1/sqrt(2)。

**逐题核查**：原答三项写成0.000174663256/0.000698653024/0.000349326512，存在小但真实算术偏差；warmup^-1.5也写成3.952847e-6，正确约3.952847e-6确实为该量，最终乘积不准。保留原答，以脚本精算值纠正，比例正确。


### D4：预热后是否按step倒数或指数下降？文中日程能否证明4000对全部模型最优？硬件时间与step数如何区分？

**结论**：预热后是s^-1/2幂律；4000不是普适最优证明。

**论文原文**：物理页7，P7:2–13。保留前后句如下：

```text
We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using
the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We
trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the
bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps
(3.5 days).
5.3 Optimizer
We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ϵ = 10 −9. We varied the learning
rate over the course of training, according to the formula:
lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5)) (3)
This corresponds to increasing the learning rate linearly for the first warmup_steps training steps,
and decreasing it thereafter proportionally to the inverse square root of the step number. We used
warmup_steps = 4000.
```

**模型原答（原样）**：

```text
预热后是按步数的逆平方根下降（不是指数衰减）:
  - 文献明确表述（给定证据）为“decreasing it thereafter proportionally to the inverse square root of the step number.”，并且公式在预热后取 step_num^{-0.5}，因此学习率随 step 以幂律（∝ step^{-1/2}）下降，而不是按照指数（e.g. e^{-k t}）下降。
关于 4000 是否对所有模型“最优”：
  - 文献在给定片段只陈述 authors 使用了 warmup_steps = 4000（“We used warmup_steps = 4000.”），但并未在该片段提供证明或实验对比以证明 4000 对所有模型或所有配置是最优的。因此无法从这段证据中证明 4000 对所有模型都是最优选择；这属于未被证据证明的结论。
硬件时间与 step 数如何区分（依据给定证据的说明与数值换算）：
  - 文献给出每一步的耗时示例：对 base 模型每个 training step 大约 0.4 秒，对 big 模型 step time 是 1.0 秒（见证据）。因此“4000 steps”在不同硬件/模型上的实际 wall-clock 时间不同：
    • 对 base 模型（0.4s/step），4000 steps ≈ 0.4 * 4000 = 1600 秒 ≈ 26.7 分钟；
    • 对 big 模型（1.0s/step），4000 steps ≈ 1.0 * 4000 = 4000 秒 ≈ 66.7 分钟。
  - 因此文中的学习率日程以 step 数为单位（warmup_steps = 4000 步），而不是以固定的 wall-clock 时间；不同模型或硬件的每步耗时会导致同样的步数对应不同的实际训练时间，但调度规则仍按步数触发（即 schedule 以 step 为尺度）。
限制：上述关于时间换算的数值基于文献中给出的每步耗时示例（base 模型约 0.4s/step，big 模型约 1.0s/step）；文献片段未给出对 4000 作为普适最优值的证据或更多硬件配置的详细比较，故不能断言其普适性或在其他硬件/模型上的最佳性。 

模型 limitations字段：文献片段仅说明了调度形式（线性预热到 warmup_steps，随后按 step^{-1/2} 衰减）并给出具体使用的 warmup_steps=4000 与示例每步耗时；片段中没有就 4000 的普适最优性提供实验对比或统计证明，且不同硬件/模型的每步耗时会影响 wall-clock 时间，但调度本身以 step 为单位。
```

**作者推导**：base约0.4秒/step，4000步粗算1600秒；big约1秒/step，粗算4000秒。文中100K步约12h、300K步约3.5days均为报告近似，不能凭粗算断言论文错误，且有运行开销。

**逐题核查**：原答单位、时间换算、最优性边界正确。

## 离线可复算算例

运行 `python offline_calc.py`，仅使用Python标准库，无网络、无GPU、不训练模型。以下为实际脚本输出；论文事实、题设教学输入与代数推导在各题中已区分。

```json
{
  "teaching_only": true,
  "unit": "learning rate per optimizer step",
  "rows": [
    {
      "step": 1000,
      "lr": 0.0001746928107421711,
      "relative_peak": 0.25000000000000006,
      "lr_model1024": 0.00012352647110032734
    },
    {
      "step": 4000,
      "lr": 0.0006987712429686843,
      "relative_peak": 1.0,
      "lr_model1024": 0.0004941058844013092
    },
    {
      "step": 16000,
      "lr": 0.00034938562148434214,
      "relative_peak": 0.5,
      "lr_model1024": 0.0002470529422006546
    }
  ],
  "width_ratio": 0.7071067811865475,
  "base_4000_steps_seconds_approx": 1600.0,
  "big_4000_steps_seconds_approx": 4000
}
```

源代码：[offline_calc.py](offline_calc.py)，保存输出：[offline-output.txt](offline-output.txt)。检查归一化、分母、比较方向的assert会在错误时中止；不是论文性能复现。

## 作者复核清单与局限

- 已按原页图核对公式、指数、转置、定义和上下文；页图见下方索引。文本抽取的断行公式已在调用前重建，保留原始冻结文件，事后不改模型输入或回答。
- 分母：逐题核对softmax归一化/平方根/宽度预算/交叉熵等对应分母；单位：参数个数、step、秒、nats、无量纲概率分开使用。比较方向已按各题算例核算。
- precision/recall、TP/FP/FN：本任务不是二分类检测，无混淆矩阵可报告；第75题的BLEU背景也不能替换成TP/FP/FN。未制造这些分数。
- 表头：本次关键证据为公式与正文，不从无表头行臆测数据；页图可能含其他表格，其数值未作为本题实测。
- 因果：单次调用、无对照、非盲、作者熟悉论文；不声称提示词带来提升，不用这批平台总分证明提示收益。
- 核查由Codex辅助执行逐项语义复核，不能冒称有独立人类审稿人或用户本人已人工签字。仍建议使用者在科研复用前复核关键推导。
- 所选片段可能不含完整定义；作者补充与题设教学假定明确标记。服务端seed/temperature和版本快照不可固定，重新调用可能给不同答案；离线脚本的确定性算术可直接复算。

## 复做步骤

1. 下载公开仓库；根README即完整正文，离线可读。按[SOURCES.md](SOURCES.md)获取同版论文，对照原页图和[numbered.txt](numbered.txt)。
2. `python validate.py sources.json output.json` 重放机械验收。第73题应保留失败字段；失败不是要求修写原答的理由。
3. `python offline_calc.py` 重放教学算术，对照保存输出。
4. 如需新真实调用，在已安装并登录Bohrium CLI的环境运行 `python reproduce_call.py --run-name my-new-run`；会产生新付费/额度调用，默认不会运行，已有目录禁止覆盖。结果是新的随机样本，不覆盖本次记录。
5. 参照[audit.md](audit.md)逐项复核原文支持性，区分论文值、教学推导和模型陈述。

## 材料与署名

原文证据：[sources.json](sources.json)、[numbered.txt](numbered.txt)。页图：[物理页7](page7.png)。完整输入输出、方法、审计与脚本均位于仓库根目录。

论文归Vaswani等；arXiv:1706.03762v7，NeurIPS 2017。本分享的原创解释与脚本MIT；论文片段版权归原作者，仅作为引用证据，不重新分发整篇PDF。外部指标定义如BLEU见[SOURCES.md](SOURCES.md)。
