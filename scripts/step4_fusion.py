"""
Step 4 — 跨文件数据融合（考核项2-④，10%）

考核要求：
- 合规构造情绪倾诉 + 索要故事/笑话的多轮问答样本
- 对话逻辑通顺

融合策略：
  第1轮 — user：情感倾诉（取自清洗后咨询数据）
  第2轮 — assistant：共情回复 + 提出讲个故事/笑话
  第3轮 — user：表示想听
  第4轮 — assistant：讲述匹配类别（焦虑→暖心故事，低落→解压笑话）的故事/笑话
"""
import json
import os
import random
import sys
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 融合对话数上限
MAX_FUSED_DIALOGUES = 2000

# 情感类别映射：从咨询数据中识别情感倾向 → 匹配的故事/笑话类别
EMOTION_KEYWORDS = {
    "焦虑": ["焦虑", "紧张", "担心", "害怕", "恐慌", "不安", "失眠", "压力"],
    "抑郁": ["抑郁", "低落", "难过", "伤心", "哭", "绝望", "没意思", "空虚"],
    "人际": ["孤独", "朋友", "同事", "关系", "沟通", "社交", "合群"],
    "家庭": ["父母", "孩子", "家庭", "婚姻", "老公", "老婆", "离婚", "吵架"],
    "职场": ["工作", "辞职", "老板", "同事", "升职", "压力", "加班"],
    "学业": ["考试", "学习", "考研", "成绩", "上学", "毕业", "论文"],
}

# 情感 → 推荐的故事/笑话类别映射
EMOTION_TO_ENTERTAINMENT = {
    "焦虑": {"category": "笑话", "reason": "让你放松一下，笑一笑"},
    "抑郁": {"category": "故事", "reason": "给你讲个温暖的故事"},
    "人际": {"category": "故事", "reason": "分享一个关于友情的故事"},
    "家庭": {"category": "故事", "reason": "讲个关于家人的温暖故事"},
    "职场": {"category": "笑话", "reason": "给你讲个职场笑话解解压"},
    "学业": {"category": "笑话", "reason": "讲个轻松的笑话放松一下"},
}


def print_divider(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def detect_emotion(text):
    """基于关键词检测文本情感倾向"""
    text_lower = text.lower()
    scores = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[emotion] = score
    if not scores:
        return "日常"
    return max(scores, key=scores.get)


def load_cleaned_data():
    """加载清洗后的咨询数据"""
    fpath = os.path.join(BASE_DIR, "data", "cleaned", "cleaned_data.json")
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ 已加载清洗后咨询数据：{len(data)} 条记录")
    return data


def load_classified_jokes():
    """加载分类后的故事/笑话数据"""
    fpath = os.path.join(BASE_DIR, "data", "classified", "joke_story_classified.json")
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ 已加载分类后故事/笑话数据：{len(data)} 条")
    return data


def extract_first_qa(counseling_record):
    """从心理咨询记录中提取首轮情感倾诉和共情回复"""
    title = counseling_record.get("question_title", "")
    content = counseling_record.get("question_content", "")
    question_text = f"{title} {content}".strip()

    if not question_text:
        return None, None

    # 找到第一个 assistant 回复作为共情回复
    first_assistant_reply = None
    for answer in counseling_record.get("answers", []):
        for dialog in answer.get("dialogs", []):
            role = dialog.get("role", "")
            dialog_content = dialog.get("content", "")
            if "answer_" in role and dialog_content:
                first_assistant_reply = dialog_content
                break
        if first_assistant_reply:
            break

    return question_text, first_assistant_reply


def generate_empathic_reply(emotion):
    """根据情感生成共情回复模板（多样化）"""
    templates = {
        "焦虑": [
            "我感受到你现在很焦虑，这种不安的感觉一定很不好受。想听听轻松的笑话放松一下吗？",
            "焦虑的时候，给自己一个喘息的 moment 很重要。要不要听个笑话转换下心情？",
        ],
        "抑郁": [
            "我能感受到你心里的沉重，你愿意说出来已经很勇敢了。让我给你讲个温暖的故事吧。",
            "低落的时候，也许一个暖心的故事能让你感觉好一些。想听听吗？",
        ],
        "人际": [
            "孤独感确实让人难受。让我给你讲个关于友情的故事吧。",
            "人和人之间的缘分有时就是这样奇妙，分享一个关于相遇的故事给你。",
        ],
        "家庭": [
            "家是我们最深的牵绊。让我给你讲个关于家人的温暖故事吧。",
            "家庭关系有时复杂但充满爱，分享一个温暖的家庭故事给你。",
        ],
        "职场": [
            "工作上的压力确实让人喘不过气。要不要听个职场笑话解解压？",
            "职场如战场，偶尔也需要笑一笑。给你讲个有意思的职场段子。",
        ],
        "学业": [
            "学业压力大的时候，别忘了给自己一点放松的时间。讲个轻松的笑话给你听。",
            "学习累了就休息一下，给你讲个有趣的笑话换换脑子。",
        ],
        "日常": [
            "生活总有起起落落，来看看这个有趣的故事怎么样？",
            "分享一个轻松的小故事给你，希望能让你会心一笑。",
        ],
    }
    return random.choice(templates.get(emotion, templates["日常"]))


def generate_fused_dialogue(question_text, empathic_reply, story_item, emotion):
    """构造完整的情绪倾诉+索要故事/笑话的多轮对话"""
    story_text = story_item.get("text", "") or story_item.get("content", "")
    category = story_item.get("category", "其他")

    # 第二轮：共情 + 提议
    if empathic_reply:
        empathy = empathic_reply[:150]
    else:
        empathy = generate_empathic_reply(emotion)

    # 区分故事/笑话的推荐语
    if category == "笑话":
        reason = "听起来不错，讲给我听听吧"
        ask_reason = "笑一笑心情会好一些"
    else:
        reason = "好啊，我很想听"
        ask_reason = "温暖的故事最能治愈人心了"

    dialogue = {
        "emotion": emotion,
        "category": category,
        "conversations": [
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": f"{empathy} 要听个{category}吗？{ask_reason}"},
            {"role": "user", "content": reason},
            {"role": "assistant", "content": story_text},
        ]
    }
    return dialogue


def main():
    random.seed(42)
    print("=" * 60)
    print("  Step 4 — 跨文件数据融合")
    print("  考核项(2)-④ 10%：情绪倾诉+索要故事/笑话多轮对话")
    print("=" * 60)

    # ========== 加载数据 ==========
    counseling_data = load_cleaned_data()
    classified_data = load_classified_jokes()

    # 按类别分组故事/笑话数据
    stories_pool = [item for item in classified_data if item.get("category") == "故事"]
    jokes_pool = [item for item in classified_data if item.get("category") == "笑话"]
    print(f"  📚 故事池：{len(stories_pool)} 条")
    print(f"  😂 笑话池：{len(jokes_pool)} 条")

    if not stories_pool and not jokes_pool:
        print("❌ 没有分类后的故事/笑话数据，请先运行 Step 3")
        return

    # ========== 执行融合 ==========
    print_divider("正在构造融合多轮对话...")
    fused_dialogues = []
    fused_count_by_emotion = {}
    skipped_no_story = 0
    skipped_no_question = 0

    for idx, record in enumerate(counseling_data):
        if len(fused_dialogues) >= MAX_FUSED_DIALOGUES:
            break

        # 提取情感倾诉和回复
        question_text, empathic_reply = extract_first_qa(record)
        if not question_text:
            skipped_no_question += 1
            continue

        # 检测情感
        emotion = detect_emotion(question_text)

        # 根据情感选择匹配的故事/笑话
        if emotion in EMOTION_TO_ENTERTAINMENT:
            target_cat = EMOTION_TO_ENTERTAINMENT[emotion]["category"]
        else:
            target_cat = random.choice(["故事", "笑话"])

        # 从对应池中随机选一条
        pool = stories_pool if target_cat == "故事" else jokes_pool
        if not pool:
            skipped_no_story += 1
            continue

        story_item = random.choice(pool)

        # 构造融合对话
        dialogue = generate_fused_dialogue(question_text, empathic_reply, story_item, emotion)
        fused_dialogues.append(dialogue)

        # 统计
        fused_count_by_emotion[emotion] = fused_count_by_emotion.get(emotion, 0) + 1

        if (idx + 1) % 500 == 0:
            print(f"  已处理 {idx + 1} 条咨询记录，生成 {len(fused_dialogues)} 条融合对话...")

    # ========== 打印统计 ==========
    print_divider("融合统计")
    print(f"  {'指标':<30} {'数量':<10}")
    print(f"  {'-' * 40}")
    print(f"  {'处理咨询记录':<30} {len(counseling_data):<10,}")
    print(f"  {'跳过-无有效问题':<30} {skipped_no_question:<10,}")
    print(f"  {'跳过-无匹配故事/笑话':<30} {skipped_no_story:<10,}")
    print(f"  {'生成融合对话数':<30} {len(fused_dialogues):<10,}")

    if fused_count_by_emotion:
        print(f"\n  情感分布：")
        for emotion, count in sorted(fused_count_by_emotion.items(), key=lambda x: -x[1]):
            pct = count / len(fused_dialogues) * 100
            print(f"    {emotion:<12} {count:<8,} ({pct:.1f}%)")

    # ========== 打印融合示例 ==========
    print_divider("融合对话示例（3条）")
    for i, d in enumerate(fused_dialogues[:3]):
        print(f"\n  📌 示例 {i + 1}（情感：{d['emotion']} → 推荐{d['category']}）")
        print(f"  {'=' * 50}")
        for j, turn in enumerate(d["conversations"]):
            role_icon = "👤" if turn["role"] == "user" else "🤖"
            content_preview = turn["content"][:100] + "..." if len(turn["content"]) > 100 else turn["content"]
            print(f"  {role_icon} [{turn['role']}]: {content_preview}")
        print(f"  {'=' * 50}")

    # ========== 保存融合数据 ==========
    output_path = os.path.join(BASE_DIR, "data", "fused", "fused_dialogue.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fused_dialogues, f, ensure_ascii=False, indent=2)

    print(f"\n💾 已保存融合对话：{output_path}")
    print(f"   共 {len(fused_dialogues)} 条多轮对话")
    print(f"\n✅ Step 4 完成。建议截图：融合统计 + 3条对话示例")


if __name__ == "__main__":
    main()
