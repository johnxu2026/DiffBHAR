# 实验记录（BHAR 自己的流程，PAMAP2 + TinyHAR）

所有实验都走 BHAR 原版的 finetune 和测试流程，和 BHAR 比较是公平的：数据、序列、分类器结构、finetune 设置、评估都一样，唯一不同的是每个传感器的发包策略 θ 从哪来。

- 数据：PAMAP2，手/胸/脚踝 3 个加速度计，12 类动作，8 个受试者留一法（LOSO）。
- 每组 3 个 seed（0、1、2）。结果是先对 8 个受试者取平均，再对 3 个 seed 取均值 ± 标准差。
- 指标：测试集准确率、宏 F1、每秒每个传感器的发包数（越少越省电）。
- 规则：传感器在距离上次发包的步数 Δ ≥ τ 时才发，τ = θ_e·(1−电量)·充电速度 + θ_p；3 个传感器各一对 (θ_e, θ_p)。θ_p 的单位是步（25 步 = 1 秒）。

## 怎么跑、怎么看

在服务器上（`~/research/DiffBHAR`，先 `source external/BHAR_codes_artifact/.venv/bin/activate`）：

```
# 1. 学 θ（写到 TAG 对应的文件夹），2. BHAR 用这个 θ finetune 分类器并测试
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 nohup sh -c "python scripts/bhar_e2e.py --seed S <选项> --device cuda --tag TAG/e2e && bash scripts/bhar_posttrain.sh conservative S scratch TAG/e2e" > /dev/null 2>&1 &

python scripts/progress.py TAG/e2e                     # 进度
python scripts/bhar_cached.py --rerun                  # 准确率 / F1 汇总表
OMP_NUM_THREADS=4 python scripts/bhar_packets.py --tags TAG/e2e   # 发包数（约 10 分钟）
```

注意：重跑时一定用新的 TAG。旧 TAG 下已有 finetune 好的分类器，BHAR 会直接拿来用，不重新训练。

## 全部结果

下表中，「起点」指 θ 训练从哪开始；「分类器」指训练 θ 时给 θ 打分用的分类器（最终测试永远是 BHAR 用最终 θ 重新 finetune 的分类器）。— 表示没测。

### 基线

| TAG | 是什么 | 准确率 | F1 | 发包/秒 |
|---|---|---|---|---|
| （无）opportunistic | 有电就发，1 个 seed | .510 | .454 | .261 |
| （无）conservative | BHAR 自己（pattern search 搜 θ，我们从头复现） | .573 ± .004 | .522 | .196 |

### 固定规则（θ_e = 0，只设 θ_p，不训练，`--epochs 0`）

| TAG | θ_p | 准确率 | F1 | 发包/秒 |
|---|---|---|---|---|
| fix50_notrain | 50 | .581 ± .007 | .534 | .213 |
| fix100_notrain | 100 | .574 ± .005 | .532 | .137 |
| fix150_notrain | 150 | .548 | .493 | .102 |
| fix200_notrain | 200 | .521 | .467 | .082 |
| fix300_notrain | 300 | .450 ± .008 | .404 | .057 |

（fix150 / fix200 的准确率是 2 个 seed 时的数，第 3 个 seed 已补跑，汇总表需要刷新。）

### 梯度训练 θ，从 BHAR 的 θ 出发（`--init bhar`，分类器 = BHAR finetune 过的）

| TAG | 选项 | 准确率 | F1 | 发包/秒 |
|---|---|---|---|---|
| e2e | 默认 | .575 ± .011 | .525 | .194 |
| pkt1 | `--pkt 1` | .581 ± .007 | .530 | .183 |
| pkt3 | `--pkt 3` | .581 ± .004 | .525 | .179 |
| pkt10 | `--pkt 10` | .578 ± .004 | .525 | .183 |
| pkt30 | `--pkt 30` | .580 ± .005 | .532 | .182 |
| neural | `--policy neural`（模仿有 bug） | .497 ± .031 | .455 | .164 |
| neural2 | `--policy neural`（修好模仿后） | .578 ± .007 | .530 | .207 |

pkt 这四组用的是旧的选择标准（val_acc − w·发包，单位不一致，2026-10-25 已改）。

### 梯度训练 θ，从 (0, 0) 出发（`--init zero`，分类器 = 预训练）

| TAG | 选项 | 准确率 | F1 | 发包/秒 |
|---|---|---|---|---|
| zero | 默认 | .556 ± .012 | .505 | .242 |
| zero_s50 | `--scale 50` | .557 ± .012 | .505 | .238 |
| joint_s50 | `--scale 50 --clf_lr 3e-4 --save_clf`（θ 和分类器一起训，BHAR 直接用我们的分类器） | .440 ± .022 | .389 | — |
| jointft_s50 | `--scale 50 --clf_lr 3e-4`（同上，但 BHAR 重新 finetune） | .571 ± .007 | .520 | .230 |
| rf_s20 | `--ste 0 --rf 1 --scale 20`（REINFORCE） | .540 ± .013 | .483 | .256 |
| rf_s5 | `--ste 0 --rf 1 --scale 5`（REINFORCE） | .531 ± .020 | .480 | .252 |
| alt | `--rounds 3 --epochs 4`（分类器和 θ 交替） | .555 ± .010 | .501 | — |
| altr | `--rounds 3 --epochs 4 --rand_ft 4` | .548 ± .003 | .503 | — |
| altr6 | `--rounds 6 --epochs 2 --rand_ft 4` | .550 ± .007 | .494 | — |

### 梯度训练 θ，从固定 θ = (0, 100) 出发（`--init zero --init_theta 0 100`，不用 BHAR）

| TAG | 选项 | 准确率 | F1 | 发包/秒 |
|---|---|---|---|---|
| fix100 | `--pkt 3`（旧选择标准） | .574 ± .005 | .532 | .137 |
| fix100r | `--pkt 3 --rounds 3 --epochs 4 --rand_ft 4`（旧选择标准） | .575 ± .002 | .532 | .136 |
| g100p0 | 默认 | .577 ± .008 | .535 | .139 |
| g100p1 | `--pkt 1` | .577 ± .007 | .535 | .150 |
| uni8 | `--pkt 1 --rounds 1 --epochs 8 --rand_ft 8`（先用 8 组随机 θ finetune 一个不偏心的分类器） | .574 ± .010 | .524 | .159 |

### 正在跑（2026-10-25）

| TAG | 选项 | 想回答的问题 |
|---|---|---|
| neural3 | `--init zero --init_theta 0 100 --policy neural --pkt 1` | 神经网络策略能不能落在固定规则曲线上方 |
| g200p0 | `--init zero --init_theta 0 200` | 从差的起点（固定规则 .521）出发，梯度能不能自己爬回 .57 |
| g300p0 | `--init zero --init_theta 0 300` | 同上，起点更差（.450） |

## 结论（到 2026-10-25 为止）

1. **从 (0, 0) 出发都失败。** τ ≤ 0 时规则等于有电就发，θ 小幅变化不改变任何发包，梯度为 0（平地）。换 REINFORCE、交替训练也没救。
2. **关键发现：打分的分类器有偏。** 冻住的预训练分类器偏爱多发包，BHAR finetune 过的分类器偏爱 BHAR 的 θ（`scripts/bhar_val_check.py`）。θ 的好坏取决于用哪个分类器打分。
3. **BHAR 在固定规则曲线下方。** 固定「至少隔 100 步才发」和 BHAR 准确率一样（.574 对 .573），F1 高 1 个点，发包少 30%。BHAR 的搜索只看准确率，不看发包；而且搜索时用单传感器分类器打分，和最终的三传感器模型不一致。
4. **梯度方法（2 参数规则）能到达这条曲线，但没有超过它。** 不用 BHAR，从 (0, 100) 出发的 g100p0：.577 / F1 .535 / .139 发包，对 BHAR 是准确率持平、发包少 29%；但和同发包的固定规则（约 .574）相比在误差内。2 参数规则的能力基本到顶。
5. **要真正超过曲线，需要更强的策略**（神经网络 + 省包惩罚，neural3），以及证明梯度不依赖好起点（g200p0、g300p0）。

## 代码改动对应

| PR | 内容 |
|---|---|
| #1 | BHAR 复现、bhar_e2e.py（e2e / joint / REINFORCE / 交替）、progress.py |
| #2 | `--pkt` 省包惩罚 |
| #3 | `--policy neural`、`scripts/bhar_hook.py`（让 BHAR 能评估神经网络策略） |
| #4 | 神经网络模仿：发/不发等权重 + DAgger |
| #5 | `--init_theta` |
| #6 | `--pkt` 时用「验证集损失 + w·发包」选模型（和训练目标一致） |
