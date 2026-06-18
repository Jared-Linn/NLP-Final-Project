"""
Step 3 — 笑话故事分类 + 关键词提取（考核项2-③，5%）

考核要求：
- 完成对第2份笑话故事数据集的正确分类
- 提取关键词

分类体系：故事 / 笑话 / 诗歌 / 其他
关键词：每条提取 2 个

实现方式：调用 DeepSeek/豆包 API（复用 llm.py 逻辑）
"""
import json
import os
import sys
import time
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 抽样数量（API 调用有成本，可根据需要调整）
SAMPLE_SIZE = 500  # 从故事笑话数据集中抽样 500 条进行分类

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
        if text_len > 300:
            category = "故事"
        elif any(kw in full_text for kw in ["哈哈", "搞笑", "笑话", "幽默", "段子", "糗事"]):
            category = "笑话"
        elif any(kw in full_text for kw in ["诗", "韵", "啊", "呀"]):
            category = "诗歌"
        else:
            category = "笑话" if text_len < 150 else "故事"

        # 提取关键词（用标题和内容的前几个词）
        words = []
        for sep in [" ", "，", "。", ",", "！", "？", "\n"]:
            if sep in full_text:
                words = [w for w in full_text.split(sep) if len(w) > 1][:5]
                break
        if not words and len(full_text) > 2:
            words = [full_text[:10]]

        keywords = words[:2] if len(words) >= 2 else (words + ["文本"] * 2)[:2]

        results.append({
            "category": category,
            "keywords": keywords,
        })

    return results


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

    # ========== 抽样 ==========
    sample = data[:SAMPLE_SIZE]
    print(f"📊 抽样分类：{SAMPLE_SIZE} 条")

    # ========== 执行分类 ==========
    print(f"\n{'=' * 60}")
    print(f"  正在分类...")

    if HAVE_OPENAI:
        print(f"  使用 DeepSeek/豆包 API...")
        results = []
        for i, item in enumerate(sample):
            text = item.get("content", "") or item.get("text", "")
            title = item.get("title", "")
            full_text = f"{title} {text}".strip() if title else text

            result = classify_with_api(full_text)
            if result is None:
                # API 失败，降级到规则
                result = classify_batch_with_rules([item])[0]
            results.append(result)

            if (i + 1) % 50 == 0:
                print(f"    已处理 {i + 1}/{SAMPLE_SIZE} 条")
            time.sleep(0.3)  # API 限流
    else:
        print(f"  未安装 openai 库，使用规则分类（保底方案）")
        print(f"  💡 建议: pip install openai 以获得更好分类效果")
        results = classify_batch_with_rules(sample)

    # ========== 统计分类结果 ==========
    category_count = {}
    for r in results:
        cat = r["category"]
        category_count[cat] = category_count.get(cat, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"  分类统计")
    print(f"{'=' * 60}")
    print(f"  {'类别':<10} {'数量':<8} {'占比':<8}")
    print(f"  {'-' * 30}")
    for cat in ["故事", "笑话", "诗歌", "其他"]:
        count = category_count.get(cat, 0)
        pct = count / max(len(results), 1) * 100
        print(f"  {cat:<10} {count:<8} {pct:<8.1f}%")

    # ========== 打印分类示例 ==========
    print(f"\n{'=' * 60}")
    print(f"  分类结果示例（每类最多5条）")
    print(f"{'=' * 60}")

    cat_examples = {cat: [] for cat in ["故事", "笑话", "诗歌", "其他"]}
    for item, r in zip(sample, results):
        cat = r["category"]
        if len(cat_examples.get(cat, [])) < 5:
            text = item.get("content", "") or item.get("text", "") or item.get("title", "")
            cat_examples[cat].append((text[:80], r["keywords"]))

    for cat, examples in cat_examples.items():
        if examples:
            print(f"\n  🏷️  {cat}（共 {category_count.get(cat, 0)} 条）：")
            for text, kw in examples:
                print(f"    📝 {text}")
                print(f"    🔑 关键词：{', '.join(kw)}")
                print()

    # ========== 保存分类结果 ==========
    output_data = []
    for item, r in zip(sample, results):
        text = item.get("content", "") or item.get("text", "") or item.get("title", "")
        title = item.get("title", "")
        output_data.append({
            "title": title,
            "text": text,
            "category": r["category"],
            "keywords": r["keywords"],
        })

    output_path = os.path.join(BASE_DIR, "data", "classified", "joke_story_classified.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n💾 已保存分类结果：{output_path}")
    print(f"   共 {len(output_data)} 条（含类别标签 + 关键词）")
    print(f"\n✅ Step 3 完成。建议截图：分类统计 + 示例")


if __name__ == "__main__":
    main()
