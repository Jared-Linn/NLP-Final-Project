# 🎓 自然语言处理期末项目：基于Qwen3.5的情感问答系统

**考核内容**：基于Qwen3.5的情感问答系统设计与实现  
**模型**：Qwen3.5-0.8B（本地路径：`./Qwen3.5-0.8B/`）  
**微调方法**：LoRA 低秩适配  
**服务**：FastAPI 多轮对话 Web 服务

---

## 📋 目录结构

```
自然语言处理期末/
│
├── Qwen3.5-0.8B/                     # 本地 Qwen3.5 模型（1.6GB）
├── data/
│   ├── No-1.json ~ No37.json         # 情感咨询补充数据（psy525来源）
│   ├── jiandanxinli_qa_data_v1.0.json # 数据集1：情感问答（~80MB）
│   ├── joke_story_v0.1.json           # 数据集2：故事/笑话（~194MB）
│   ├── cleaned/                       # ② 清洗后数据输出
│   ├── classified/                    # ③ 分类+关键词输出
│   ├── fused/                         # ④ 融合多轮对话输出
│   └── split/                         # ⑤ train/test 划分
│
├── scripts/                           # 全流程分步脚本
│   ├── step1_load_data.py             # ① 多源语料读取（5%）
│   ├── step2_clean_data.py            # ② 脏数据清洗（3%）
│   ├── step3_classify.py              # ③ 分类+关键词（5%）
│   ├── step4_fusion.py                # ④ 跨文件融合（10%）
│   ├── step5_tokenize_split.py        # ⑤ 分词+划分（5%）
│   ├── step6_setup_lora.py            # ⑥ 模型搭建+LoRA（15%）
│   ├── step7_train_test.py            # ⑦ 训练+测试（10%）
│   ├── step8_evaluate.py              # ⑧ 模型评价（5%）
│   └── run_all.py                     # 一键全流程执行
│
├── app/                               # Web 服务（待完成）
│   └── templates/
│
├── outputs/
│   ├── lora_adapter/                  # LoRA 微调权重（55MB）
│   │   ├── checkpoint-76/             # 第76步checkpoint（含优化器状态）
│   │   └── lora_adapter/              # 最终adapter权重
│   ├── inference_test_results.json    # 推理测试结果（5KB）
│   └── evaluation_report.json         # 模型评价报告（22KB）
│
├── 计划表.md                          # 考核项↔文件映射追踪
├── requirements.txt                   # Python 依赖
├── .gitignore                         # Git 忽略规则
├── setup.py                           # 项目安装配置
└── README.md                          # 本文档
```

---

## 🧩 考核项与模块对应

| 考核项 | 分值 | 脚本 | 功能简述 |
|--------|------|------|----------|
| (1) 多源语料读取 | **5%** | `step1_load_data.py` | 读取两份数据集，识别情感问答/故事/笑话三类 |
| (2)-① 分别加载两份语料 | **2%** | `step1_load_data.py` | 加载 jiandanxinli + joke_story |
| (2)-② 脏数据清洗 | **3%** | `step2_clean_data.py` | 去广告/无效社交/去重（30877→24428条） |
| (2)-③ 分类+关键词 | **5%** | `step3_classify.py` | 对故事笑话分类（故事/笑话/诗歌/其他）+ 关键词提取 |
| (2)-④ 跨文件融合 | **10%** | `step4_fusion.py` | 情感倾诉+索要故事/笑话的多轮对话（2000条） |
| (2)-⑤ 分词+划分 | **5%** | `step5_tokenize_split.py` | Qwen3 tokenizer编码 + ChatML格式 + 8:2划分 |
| (3) 模型搭建 | **15%** | `step6_setup_lora.py` | GPU/CPU检测 + LoRA配置 + 超参定义 |
| (4) 训练+测试 | **10%** | `step7_train_test.py` | LoRA微调 + loss收敛 + 多轮对话功能测试 |
| (5) 模型评价 | **5%** | `step8_evaluate.py` | 微调前后对比（连贯性/共情度/适配性） |
| (6) 论文格式 | **20%** | - | 报告文档 + 过程截图 |

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Windows / Linux / macOS
- 最低 16GB RAM（CPU 推理）
- 推荐：NVIDIA GPU + 8GB+ 显存

### 安装依赖

```bash
pip install -r requirements.txt
```

### 全流程执行

```bash
python scripts/run_all.py
```

或分步执行（每步可截图作为报告素材）：

```bash
python scripts/step1_load_data.py     # [1] 多源读取
python scripts/step2_clean_data.py    # [2] 脏数据清洗
python scripts/step3_classify.py      # [3] 分类+关键词
python scripts/step4_fusion.py        # [4] 跨文件融合
python scripts/step5_tokenize_split.py# [5] 分词+划分
python scripts/step6_setup_lora.py    # [6] 模型搭建
python scripts/step7_train_test.py    # [7] 训练+测试
python scripts/step8_evaluate.py      # [8] 模型评价
```

### 云端训练 (RTX 3090)
1. SSH 连接到 GPU 服务器
2. 上传项目代码
3. 下载模型: `python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-0.8B')"`
4. 运行: `python scripts/run_all.py`

---

## 🔬 数据处理流水线详情

### Step 1 — 多源语料读取

加载两份独立数据集 + 补充数据，识别三类数据：

| 数据集 | 来源 | 类型 | 条数 |
|--------|------|------|------|
| jiandanxinli_qa_data_v1.0.json | 简单心理 | 情感问答 | 30,877 |
| joke_story_v0.1.json | 互联网 | 故事+笑话 | 166,970 |
| No-1~37.json | psy525 | 情感问答 | ~260,000 |

### Step 2 — 脏数据清洗

清洗规则：
- **广告引流**：`加V|扫码|免费|http|qq\d+|微信` 等正则匹配
- **无效社交**：`在吗|睡了吗|嗯|好的|谢谢` 等无意义单轮
- **文本去重**：MD5 content hash
- **过短过滤**：<5字符丢弃

清洗效果：jiandanxinli 30,877条→**24,428条**；psy525补充数据259,338条同步清洗；两份数据合并输出

### Step 3 — 分类+关键词提取

对 `joke_story_v0.1.json` 进行规则+API混合分类，覆盖前20,000条（占全量12%），API精分每类100条验证

### Step 4 — 跨文件融合（核心）

构造四轮对话结构：

```
👤 [user]:    最近很焦虑，考试压力大...
🤖 [assistant]: 我理解你的压力。要听个笑话放松一下吗？
👤 [user]:    好啊，讲给我听听
🤖 [assistant]: [匹配的幽默故事/笑话内容...]
```

情感检测 → 类别匹配映射：

| 用户情感 | 推荐类别 | 适配理由 |
|----------|----------|----------|
| 焦虑 | 😂 笑话 | 放松心情，笑一笑 |
| 抑郁 | 📖 故事 | 温暖故事治愈 |
| 人际 | 📖 故事 | 友情相关故事 |
| 职场 | 😂 笑话 | 解压 |
| 学业 | 😂 笑话 | 轻松换脑子 |

### Step 5 — 分词 + 格式统一 + 划分

- **分词器**：Qwen3.5 tokenizer（词表248,077）
- **格式**：ChatML `<|im_start|>role\n内容<|im_end|>`
- **label掩码**：仅 `assistant` 回复参与 loss 计算
- **划分**：8:2 随机 → 1600 train / 400 test
- **token长度**：平均395，最长512（受GPU显存限制）

---

## 🏗️ 模型训练

### LoRA 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 基础模型 | Qwen3.5-0.8B | 8亿参数对话模型 |
| LoRA r | 16 | 低秩矩阵维度 |
| LoRA alpha | 32 | 缩放系数=2×r |
| LoRA dropout | 0.1 | 防过拟合 |
| 目标模块 | q_proj, k_proj, v_proj, o_proj | 注意力层 |
| 可训练参数量 | ~2.1M | 占基础模型~0.2% |

### 训练超参数

| 参数 | 值 |
|------|-----|
| 学习率 | 2e-4 |
| 训练轮数 | 3 |
| 批次大小 | 1（梯度累积4，GPU实测RTX3090 24GB） |
| 优化器 | AdamW |
| 损失函数 | CrossEntropyLoss |
| 调度器 | Cosine |
| 最大序列长度 | 512（GPU显存适配） |

### 训练方式

- **GPU**：自动检测 CUDA，使用 FP16/BF16
- **CPU**：FP32 回退，兼容无 GPU 环境

---

## 📊 模型评价

对比微调前后模型在以下维度的表现：

| 维度 | 评价方法 | 说明 |
|------|---------|------|
| 连贯性 | 对话逻辑评分（1-5） | 上下文衔接是否自然 |
| 共情度 | 情感回应评分（1-5） | 能否识别情感并温暖回应 |
| 故事/笑话适配 | 类别匹配率 | 请求类别与实际匹配度 |

> 详细评价结果见 `outputs/evaluation_report.json`

---

## 📁 数据文件大小参考

| 文件 | 大小 | 说明 | 是否提交 |
|------|------|------|----------|
| `jiandanxinli_qa_data_v1.0.json` | ~80 MB | 原始数据 | ❌ .gitignore |
| `joke_story_v0.1.json` | ~194 MB | 原始数据 | ❌ .gitignore |
| `data/No-*.json` | ~25-60 MB/个 | 原始数据 | ❌ .gitignore |
| `data/cleaned/cleaned_data.json` | ~100 MB | 清洗后合并数据 | ❌ |
| `data/classified/joke_story_classified.json` | ~12 MB | 分类+关键词 | ❌ |
| `data/fused/fused_dialogue.json` | ~4 MB | 融合多轮对话 | ❌ |
| `data/split/train_chatml.json` | ~3.3 MB | ChatML格式训练集 | ❌ |
| `outputs/lora_adapter/` | ~55 MB | 训练输出（含checkpoint） | ❌ |

*更新日期：2026/06/21 第二次完善*

---

## ✅ RTX 3090 云端训练结果 (2026/06/21)

| 指标 | 值 |
|------|-----|
| 训练设备 | NVIDIA RTX 3090 24GB (创投云) |
| 精度 | BF16 |
| 训练数据 | 1600 条（全量随机打乱） |
| 训练轮数 | 3 |
| 总步数 | 1200 |
| 等效批次 | 4 |
| 训练耗时 | ~56 分钟 (3373秒) |
| 速度 | 2.81 秒/步, 1.42 样本/秒 |
| 初始 Loss | 4.349 |
| 最终 Loss | 3.079 |
| Loss 降幅 | 29.2% |
| 负面情绪联动 | 3/3 (100%) |

**Loss 收敛**：
- Step 1: 4.35 → Step 50: 3.19 → Step 100: 3.13
- Step 200: 2.91 → Step 400: 2.72 → Step 600: 2.76
- Step 800: 2.72 → Step 1000: 2.62 → Step 1200: 2.97
- train_loss: **3.079**

**与 GTX 1060 对比**：
| 指标 | GTX 1060 3GB | RTX 3090 24GB |
|------|:---:|:---:|
| 速度 | ~50s/步 | 2.81s/步 (快18倍) |
| 训练数据 | 150条/2轮 | 1600条/3轮 |
| 耗时 | 66分钟 | 56分钟 |
| 最终Loss | 3.60 | 3.08 |

---

## 📚 技术栈

- **模型**：Qwen3.5-0.8B（通义千问）
- **微调**：PEFT + LoRA + HuggingFace Transformers
- **数据处理**：Python, JSON, Regex
- **服务**：FastAPI + Uvicorn + Jinja2
- **推理**：CPU FP32 / GPU FP16
- **外部API**：DeepSeek / 豆包（分类+关键词）

---

## 📌 注意事项

1. **数据不提交**：原始 JSON 数据文件和模型文件体积大，已在 `.gitignore` 中排除
2. **截图素材**：每步脚本都包含详细的控制台输出，可直接用于报告截图
3. **API Key**：`step3_classify.py` 中内置了测试 Key，建议改为环境变量
4. **训练耗时**：CPU 环境训练可能需要数小时，GPU 环境可大幅加速
5. **一键执行**：`python scripts/run_all.py` 自动顺序执行全部 8 个步骤
