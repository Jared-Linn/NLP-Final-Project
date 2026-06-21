"""
Step 3 — 笑话故事分类 + 关键词提取（考核项2-③，5%）

考核要求：
- 完成对第2份笑话故事数据集的正确分类
- 提取关键词

分类体系：故事 / 笑话 / 诗歌 / 其他
关键词：每条提取 2 个

实现方式：混合策略 — 规则先粗分全部 + API 精分抽样验证
"""
import json
import os
import sys
import time
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ========== 策略参数 ==========
# API 精分抽样子集大小（API 调用有成本）
SAMPLE_SIZE = 5000
# 规则分类处理上限（全量数据太大，先处理前 N 条）
RULE_PROCESS_COUNT = 20000
# 每类用 API 精分抽样数
API_SAMPLE_PER_CATEGORY = 100
# 断点续跑临时文件
CHECKPOINT_FILE = os.path.join(BASE_DIR, "data", "classified", "_step3_checkpoint.json")

# ==================== DeepSeek API 调用 ====================
try:
    from openai import OpenAI
    HAVE_OPENAI = True
except ImportError:
    HAVE_OPENAI = False

SYSTEM_PROMPT_CLASSIFY = """你是一个文本分类专家。
请对用户输入的文本进行分类，并提取关键词。

分类标签（仅选一个）：
- 故事：有情节叙述、人物、场景的完整故事
- 笑话：短小幽默、有笑点、段子
- 诗歌：有韵律、分行的文学体裁
- 其他：以上都不属于的文本

关键词：提取 2 个最能概括文本内容的关键词

返回格式（严格 JSON）：
{"category": "故事/笑话/诗歌/其他", "keywords": ["关键词1", "关键词2"]}
只返回 JSON，不要其他内容。"""


def classify_with_api(text):
    """调用 DeepSeek/豆包 API 进行文本分类和关键词提取"""
    if not HAVE_OPENAI:
        return None

    try:
        client = OpenAI(
            api_key="ark-d25f9cd7-14a5-41f9-a31c-f9f21f43eac4-8ab2f",
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )

        response = client.chat.completions.create(
            model="doubao-seed-2-0-pro-260215",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_CLASSIFY},
                {"role": "user", "content": text[:300]},  # 取前300字节省token
            ],
            stream=False,
            temperature=0.3,
            max_tokens=100,
        )
        result = response.choices[0].message.content.strip()

        # 提取 JSON
        if "{" in result:
            result = result[result.index("{"):result.rindex("}") + 1]
        return json.loads(result)
    except Exception as e:
        return {"category": "其他", "keywords": ["文本", "未分类"]}


def classify_batch_with_rules(texts):
    """基于规则的快速分类（不使用API），作为保底方案"""
    results = []
    for text in texts:
        content = text.get("content", "") or text.get("text", "")
        title = text.get("title", "")
        full_text = f"{title} {content}".strip()
        text_len = len(full_text)

        # 简单规则分类
        if text_len < 10:
            # 文本太短，归为"其他"
            category = "其他"
        elif text_len > 300:
            category = "故事"
        elif any(kw in full_text for kw in ["哈哈", "搞笑", "笑话", "幽默", "段子", "糗事"]):
            category = "笑话"
        elif any(kw in full_text for kw in ["诗", "韵", "啊", "呀"]):
            category = "诗歌"
        else:
            # 无法匹配任何特征词，归为"其他"
            category = "其他"

        # 提取关键词（用标题和内容的前几个词）
        words = []
        for sep in [" ", "，", "。", ",", "！", "？", "\n"]:
            if sep in full_text:
                words = [w.strip() for w in full_text.split(sep) if len(w.strip()) > 1][:5]
                break
        if not words and len(full_text) > 2:
            words = [full_text[:10]]

        # 取前2个词，每个词截断到最长8个字符；不足2个词用"文本"+"内容"填充
        kw1 = words[0][:8] if len(words) >= 1 else "文本"
        kw2 = words[1][:8] if len(words) >= 2 else "内容"
        keywords = [kw1, kw2]

        results.append({
            "category": category,
            "keywords": keywords,
        })

    return results


def _load_checkpoint():
    """加载断点续跑缓存，返回 {index_in_sample: result_dict}"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_checkpoint(cp):
    """保存断点续跑缓存"""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)


def main():
    print("=" * 60)
    print("  Step 3 — 笑话故事分类 + 关键词提取")
    print("  考核项(2)-③ 5%")
    print("=" * 60)

    # ========== 加载笑话故事数据集 ==========
    fpath = os.path.join(BASE_DIR, "joke_story_v0.1.json")
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = len(data)
    print(f"\n✅ 已加载故事笑话数据：{total:,} 条")

    # ========== 限长：只处理前 RULE_PROCESS_COUNT 条 ==========
    proc_data = data[:RULE_PROCESS_COUNT]
    print(f"📊 规则分类范围：前 {RULE_PROCESS_COUNT:,} 条（全量 {total:,} 条中的 {RULE_PROCESS_COUNT/total*100:.1f}%）")

    # ========== 第一阶段：规则粗分 ==========
    print(f"\n{'=' * 60}")
    print(f"  第一阶段：规则粗分（全量）...")
    rule_results = classify_batch_with_rules(proc_data)
    print(f"  ✅ 规则分类完成：{len(rule_results)} 条")
    cat_rule_count = {}
    for r in rule_results:
        cat = r["category"]
        cat_rule_count[cat] = cat_rule_count.get(cat, 0) + 1

    print(f"{'':>4}{'类别':<10} {'规则分类数':<12}")
    for cat in ["故事", "笑话", "诗歌", "其他"]:
        print(f"{'':>4}{cat:<10} {cat_rule_count.get(cat, 0):<12}")

    # ========== 第二阶段：API 精分抽样 ==========
    print(f"\n{'=' * 60}")
    print(f"  第二阶段：API 精分抽样验证...")

    # 从规则结果中为每类抽 API_SAMPLE_PER_CATEGORY 条
    cat_indices = {cat: [] for cat in ["故事", "笑话", "诗歌", "其他"]}
    for idx, r in enumerate(rule_results):
        cat = r["category"]
        if len(cat_indices[cat]) < API_SAMPLE_PER_CATEGORY:
            cat_indices[cat].append(idx)

    api_sample_indices = []
    for cat in ["故事", "笑话", "诗歌", "其他"]:
        indices = cat_indices[cat]
        api_sample_indices.extend(indices)
        print(f"  📌 {cat}：抽 {len(indices)} 条进行 API 精分")

    # 限制 API 总调用量不超过 SAMPLE_SIZE
    if len(api_sample_indices) > SAMPLE_SIZE:
        import random
        random.seed(42)
        api_sample_indices = sorted(random.sample(api_sample_indices, SAMPLE_SIZE))
    print(f"  📊 API 精分总数：{len(api_sample_indices)} 条")

    api_results_dict = {}  # {idx: result_dict}

    if HAVE_OPENAI:
        # 加载断点续跑缓存
        checkpoint = _load_checkpoint()
        if checkpoint:
            done_count = len(checkpoint)
            print(f"  🔄 检测到断点缓存：{done_count} 条已处理，继续...")
        else:
            checkpoint = {}
            done_count = 0

        print(f"  使用豆包 API...")
        for seq, idx in enumerate(api_sample_indices):
            str_idx = str(idx)
            # 跳过已处理
            if str_idx in checkpoint:
                api_results_dict[idx] = checkpoint[str_idx]
                continue

            item = proc_data[idx]
            text = item.get("content", "") or item.get("text", "")
            title = item.get("title", "")
            full_text = f"{title} {text}".strip() if title else text

            result = classify_with_api(full_text)
            if result is None:
                result = rule_results[idx]  # API 失败降级到规则
            api_results_dict[idx] = result

            # 写入断点
            checkpoint[str_idx] = result
            if (seq + 1) % 20 == 0 or (seq + 1) == len(api_sample_indices):
                _save_checkpoint(checkpoint)

            done = len(checkpoint)
            if (done) % 50 == 0 or done == len(api_sample_indices):
                print(f"    已处理 {done}/{len(api_sample_indices)} 条")
            time.sleep(0.3)  # API 限流

        # 清理断点文件（完成后）
        if len(checkpoint) >= len(api_sample_indices):
            try:
                os.remove(CHECKPOINT_FILE)
                print(f"  🧹 断点缓存已清理")
            except Exception:
                pass
    else:
        print(f"  未安装 openai 库，跳过 API 精分（仅使用规则分类）")
        print(f"  💡 建议: pip install openai 以获得更好分类效果")

    # ========== 合成最终结果 ==========
    print(f"\n{'=' * 60}")
    print(f"  合成最终分类结果...")

    # 兜底策略：如果没跑 API 或 API 结果为空，直接使用全量规则结果
    pure_rule_mode = (not HAVE_OPENAI) or (len(api_results_dict) == 0)
    if pure_rule_mode:
        print(f"  ⚠️  纯规则模式：未使用 API 精分，直接输出全量规则分类结果")
        for r in rule_results:
            r["source"] = "规则"
        output_data = []
        for idx, item in enumerate(proc_data):
            text = item.get("content", "") or item.get("text", "") or item.get("title", "")
            title = item.get("title", "")
            r = rule_results[idx]
            output_data.append({
                "title": title,
                "text": text,
                "category": r["category"],
                "keywords": r["keywords"],
                "source": "规则",
            })
        # 用纯规则结果重算分类统计
        cat_rule_count = {}
        for od in output_data:
            cat = od["category"]
            cat_rule_count[cat] = cat_rule_count.get(cat, 0) + 1
        # 清空 API 统计以示区分
        cat_api_count = {cat: 0 for cat in ["故事", "笑话", "诗歌", "其他"]}
        api_results_dict = {}

    else:
        # 混合模式：API 精分 + 规则兜底
        output_data = []
        cat_api_count = {cat: 0 for cat in ["故事", "笑话", "诗歌", "其他"]}

        for idx, item in enumerate(proc_data):
            text = item.get("content", "") or item.get("text", "") or item.get("title", "")
            title = item.get("title", "")

            if idx in api_results_dict:
                r = api_results_dict[idx]
                source = "API"
                cat_api_count[r["category"]] = cat_api_count.get(r["category"], 0) + 1
            else:
                r = rule_results[idx]
                source = "规则"

            output_data.append({
                "title": title,
                "text": text,
                "category": r["category"],
                "keywords": r["keywords"],
                "source": source,
            })

    # ========== 统计 ==========
    cat_total_count = {}
    for od in output_data:
        cat = od["category"]
        cat_total_count[cat] = cat_total_count.get(cat, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"  分类统计总表")
    print(f"{'=' * 60}")
    print(f"  {'类别':<10} {'规则分类':<10} {'API精分':<10} {'合计':<10} {'占比':<8}")
    print(f"  {'-' * 48}")
    for cat in ["故事", "笑话", "诗歌", "其他"]:
        rc = cat_rule_count.get(cat, 0)
        ac = cat_api_count.get(cat, 0)
        tc = cat_total_count.get(cat, 0)
        pct = tc / max(len(output_data), 1) * 100
        print(f"  {cat:<10} {rc:<10} {ac:<10} {tc:<10} {pct:<8.1f}%")
    print(f"  {'-' * 48}")
    print(f"  {'合计':<10} {cat_rule_count.get('故事',0)+cat_rule_count.get('笑话',0)+cat_rule_count.get('诗歌',0)+cat_rule_count.get('其他',0):<10} {cat_api_count.get('故事',0)+cat_api_count.get('笑话',0)+cat_api_count.get('诗歌',0)+cat_api_count.get('其他',0):<10} {len(output_data):<10} {100:<8.1f}%")

    # ========== 打印分类示例（API 精分类） ==========
    print(f"\n{'=' * 60}")
    print(f"  API 精分结果示例（每类最多5条）")
    print(f"{'=' * 60}")

    api_examples = {cat: [] for cat in ["故事", "笑话", "诗歌", "其他"]}
    for od in output_data:
        if od["source"] != "API":
            continue
        cat = od["category"]
        if len(api_examples.get(cat, [])) < 5:
            api_examples[cat].append((od["text"][:80], od["keywords"]))

    for cat, examples in api_examples.items():
        if examples:
            print(f"\n  🏷️  {cat}（API精分 {cat_api_count.get(cat, 0)} 条）：")
            for text, kw in examples:
                print(f"    📝 {text}")
                print(f"    🔑 关键词：{', '.join(kw)}")
                print()

    # ========== 保存分类结果 ==========
    output_path = os.path.join(BASE_DIR, "data", "classified", "joke_story_classified.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n💾 已保存分类结果：{output_path}")
    print(f"   共 {len(output_data)} 条（含类别标签 + 关键词 + 来源标记）")
    print(f"   - 规则分类：{len(output_data) - len(api_results_dict)} 条")
    print(f"   - API 精分：{len(api_results_dict)} 条")
    print(f"\n✅ Step 3 完成。建议截图：分类统计总表 + API精分示例")


if __name__ == "__main__":
    main()
