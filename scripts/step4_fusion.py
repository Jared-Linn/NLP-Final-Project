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
    """从心理咨询记录中提取首轮情感倾诉（仅提取用户问题，不再提取原始assistant回复）"""
    title = counseling_record.get("question_title", "")
    content = counseling_record.get("question_content", "")
    question_text = f"{title} {content}".strip()

    if not question_text:
        return None

    return question_text


def generate_empathic_reply(emotion):
    """根据情感生成共情回复模板（多样化，每种情感≥5个模板，含随机emoji和措辞扰动）"""
    emoji_pool = ["😊", "💪", "✨", "🌟", "💛", "🌸", "🌈", "☀️", "🍀", "🎈"]
    emoji = random.choice(emoji_pool) if random.random() < 0.6 else ""

    templates = {
        "焦虑": [
            "我感受到你现在很焦虑，这种不安的感觉一定很不好受。想听听轻松的笑话放松一下吗？",
            "别太紧张啦，给自己一个喘息的 moment 很重要。要不要听个笑话转换下心情？",
            "焦虑的时候更要照顾好自己 {}。来，给你讲个笑话，暂时忘掉烦恼吧。",
            "压力太大了对吧？这时候需要一点轻松 {}。听个笑话怎么样？",
            "紧绷的神经也需要休息呀 {}。让我讲个笑话帮你放松放松好吗？",
            "越是焦虑越要给自己留点喘息的空间，给你讲个有意思的乐呵乐呵吧 {}。",
        ],
        "抑郁": [
            "我能感受到你心里的沉重，你愿意说出来已经很勇敢了。让我给你讲个温暖的故事吧。",
            "低落的时候，也许一个暖心的故事能让你感觉好一些。想听听吗 {}？",
            "抱抱你 {}，一切都会好起来的。让我讲个温暖的小故事陪伴你吧。",
            "心里的难过我都懂，让一个暖心的故事陪陪你，好吗 {}？",
            "你并不孤单 {}。来，我给你讲个温暖的故事，希望能给你一点点力量。",
            "有时候我们需要一点点温柔 {}。让我分享一个治愈的小故事给你。",
        ],
        "人际": [
            "孤独感确实让人难受。让我给你讲个关于友情的故事吧 {}。",
            "人和人之间的缘分有时就是这样奇妙，分享一个关于相遇的故事给你。",
            "交朋友这件事急不来，但好故事总能让心里暖暖的 {}。想听一个吗？",
            "有时候我们需要打开心扉 {}。给你讲个关于缘分的小故事吧。",
            "感到孤单的时候，故事里的温暖也能陪伴你 {}。来听听这个吧。",
            "人与人之间的联结有时候就在一个故事里 {}，分享一个给你。",
        ],
        "家庭": [
            "家是我们最深的牵绊。让我给你讲个关于家人的温暖故事吧 {}。",
            "家庭关系有时复杂但充满爱，分享一个温暖的家庭故事给你 {}。",
            "说起家人，心里总是又酸又暖的 {}。给你讲个关于家的故事吧。",
            "亲情是这世上最柔软的角落 {}。来听一个关于家人的故事吧。",
            "和家人之间的点点滴滴，总能触动人心 {}。分享一个温暖的小故事。",
            "不管走多远，家永远是港湾 {}。让我讲个关于家的故事给你听。",
        ],
        "职场": [
            "工作上的压力确实让人喘不过气。要不要听个职场笑话解解压 {}？",
            "职场如战场，偶尔也需要笑一笑。给你讲个有意思的职场段子。",
            "打工人的日常太真实了 {}，来听个职场笑话放松一下吧。",
            "工作再忙也要记得开心呀 {}。给你分享一个超好笑的职场故事。",
            "职场里那些哭笑不得的事儿最解压了 {}，想不想听一个？",
            "被工作压得喘不过气？来 {}，听个职场笑话充充电。",
        ],
        "学业": [
            "学业压力大的时候，别忘了给自己一点放松的时间。讲个轻松的笑话给你听。",
            "学习累了就休息一下，给你讲个有趣的笑话换换脑子 {}。",
            "读书备考确实不容易 {}，来个小笑话让你轻松一分钟。",
            "脑子转不动的时候最适合听笑话啦 {}，我给你讲一个。",
            "学霸也需要劳逸结合 {}，来听个搞笑的段子吧。",
            "放下书本放松一下 {}，给你讲个有意思的笑话怎么样？",
        ],
        "日常": [
            "生活总有起起落落，来看看这个有趣的故事怎么样 {}？",
            "分享一个轻松的小故事给你，希望能让你会心一笑 {}。",
            "今天天气不错 {}，正好适合听个小故事放松一下。",
            "给你分享一个有趣的小故事，保证让你心情好起来 {}。",
            "日子平淡的时候，一个有趣的故事就是最好的调味剂 {}。",
            "来 {}，我这儿有个不错的小故事，要不要听听看？",
        ],
    }

    chosen = random.choice(templates.get(emotion, templates["日常"]))
    return chosen.format(emoji) if "{}" in chosen else f"{chosen}{emoji}"


# 第3轮 user 索要语（多样化，至少5种变体）
USER_ASK_PHRASES = [
    "好啊讲给我听听！",
    "想听，快讲吧。",
    "来一个吧~",
    "讲个呗！",
    "好呀好呀！",
    "嗯嗯，想听！",
    "来吧，我准备好了。",
    "说说看~",
    "可以呀，讲吧！",
    "行啊，讲来听听。",
]


# 第3轮索要语轮询计数器（全局，确保不同对话使用不同索要语）
_USER_ASK_INDEX = 0
_USER_ASK_SHUFFLED = []


def _init_user_ask_pool():
    """初始化并打乱索要语池，确保轮询使用时多样性最大化"""
    global _USER_ASK_SHUFFLED, _USER_ASK_INDEX
    if not _USER_ASK_SHUFFLED:
        _USER_ASK_SHUFFLED = USER_ASK_PHRASES.copy()
        random.shuffle(_USER_ASK_SHUFFLED)
        _USER_ASK_INDEX = 0


def _next_user_ask():
    """按轮询顺序取索要语，用完60%后重新洗牌"""
    global _USER_ASK_INDEX, _USER_ASK_SHUFFLED
    _init_user_ask_pool()
    phrase = _USER_ASK_SHUFFLED[_USER_ASK_INDEX]
    _USER_ASK_INDEX += 1
    # 用完~60%就重新洗牌，既保证多样性又避免正好卡在某个边界
    if _USER_ASK_INDEX >= len(_USER_ASK_SHUFFLED) * 0.6:
        remaining = _USER_ASK_SHUFFLED[_USER_ASK_INDEX:]
        random.shuffle(remaining)
        _USER_ASK_SHUFFLED = remaining
        _USER_ASK_INDEX = 0
    return phrase


def generate_fused_dialogue(question_text, story_item, emotion):
    """构造完整的情绪倾诉+索要故事/笑话的多轮对话

    第2轮 assistant 内容完全由 generate_empathic_reply() 生成，
    不再拼接原始 assistant 回复。

    第3轮采用轮询策略而非 random.choice，确保10种索要语均匀出现。
    """
    story_text = story_item.get("text", "") or story_item.get("content", "")
    category = story_item.get("category", "其他")

    # 第2轮：完全由共情模板生成
    empathic_reply = generate_empathic_reply(emotion)

    # 区分故事/笑话的推荐语
    if category == "笑话":
        ask_reason = "笑一笑心情会好一些"
    else:
        ask_reason = "温暖的故事最能治愈人心了"

    # 第3轮：轮询取索要语（不再是随机choice，确保10种变体全部出现）
    user_ask = _next_user_ask()

    dialogue = {
        "emotion": emotion,
        "category": category,
        "conversations": [
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": f"{empathic_reply} 要听个{category}吗？{ask_reason}"},
            {"role": "user", "content": user_ask},
            {"role": "assistant", "content": story_text},
        ]
    }
    return dialogue


# ====== 内容安全：禁止出现的不当词汇 ======
BLOCKED_KEYWORDS = [
    "口交", "乳房", "做爱", "操你", "操他", "操我", "fuck", "他妈",
    "去死", "贱人", "婊子", "鸡巴", "屁股", "上床", "性交", "阴部",
    "阴道", "阴茎", "情色", "色情", "裸体", "裸照", "强奸", "轮奸",
    "乱伦", "卖淫", "嫖娼", "淫荡", "下流", "畜生", "王八蛋",
    "黑人", "黑鬼", "尼哥", "nigger", "chink", "白痴", "脑残",
    "弱智", "变态", "裸露", "性爱", "荤段子", "黄色笑话",
    "裸聊", "一夜情", "约炮", "打炮", "骚货", "浪货",
]

# 第4轮中即使不触发关键词也判定为空洞语的模式（表情/拟声词/敷衍词）
EMPTY_RESPONSE_PATTERNS = [
    "哈哈", "呵呵", "嘿嘿", "嘻嘻", "hhh", "666", "不错",
    "好的", "嗯嗯", "可以", "行吧", "哦哦",
]


def validate_fused_dialogue(dialogue):
    """检查融合对话的基本质量，不合格返回False"""
    conversations = dialogue.get("conversations", [])
    if len(conversations) != 4:
        return False

    # 检查是否有空内容
    for turn in conversations:
        content = turn.get("content", "").strip()
        if not content:
            return False

    # 检查第2轮是否包含"故事"或"笑话"关键词（确保推荐语正常）
    round2_content = conversations[1].get("content", "")
    if "故事" not in round2_content and "笑话" not in round2_content:
        return False

    # ---- 内容安全过滤：第4轮敏感词检查 ----
    round4_content = conversations[3].get("content", "")
    round4_lower = round4_content.lower()

    # 检查第4轮是否包含不当词汇
    for kw in BLOCKED_KEYWORDS:
        if kw in round4_content or kw in round4_lower:
            print(f"    ⛔ 跳过含敏感词[{kw}]的融合对话")
            return False

    # ---- 检查第1轮和第4轮组合的安全性 ----
    round1_content = conversations[0].get("content", "")
    round1_lower = round1_content.lower()
    severe_negative_keywords = ["想死", "自杀", "活不下去", "不想活", "死了算"]
    for severe_kw in severe_negative_keywords:
        if severe_kw in round1_lower:
            # 用户有严重负面倾向时，第4轮必须是"故事"类型
            if dialogue.get("category") != "故事":
                print(f"    ⛔ 用户'{severe_kw}'但推荐了笑话，跳过（必须用暖心故事）")
                return False
            # 同时检查故事内容本身也不包含不当内容（二次防御）
            for kw in BLOCKED_KEYWORDS:
                if kw in round4_content or kw in round4_lower:
                    print(f"    ⛔ 严重用户场景下第4轮含敏感词[{kw}]，跳过")
                    return False
            break

    # 检查第4轮（故事/笑话正文）内容长度≥30字（原20字，已提高标准）
    if len(round4_content) < 30:
        return False

    # 检查第4轮是否空洞（只有哈哈/呵呵等敷衍语）
    stripped = round4_content.strip()
    # 如果去掉所有标点和空格后，剩余完全由空洞模式组成，判定为空洞
    import re
    cleaned = re.sub(r'[\s,，。！？、；：""''【】《》（）\!\?\.,;:\(\)\[\]\{\}]', '', stripped)
    if cleaned and all(cleaned == pat or cleaned.startswith(pat) or cleaned.endswith(pat) for pat in EMPTY_RESPONSE_PATTERNS):
        return False
    # 更严格的单独检查：如果第4轮真的是空洞敷衍（纯表情/纯语气词），也应拒绝
    hollow_count = sum(1 for pat in EMPTY_RESPONSE_PATTERNS if pat in cleaned)
    if hollow_count >= 2 and len(cleaned) <= 10:
        return False

    return True


def _check_emotional_adaptation(emotion, target_cat, question_text):
    """增强的情感适配检查：
    严重负面情绪（抑郁）必须配温暖故事而非笑话。
    这是安全底线——给想自杀的用户推笑话是严重违规。
    """
    severe_negative_emotions = ["抑郁"]  # 抑郁情绪必须配故事
    if emotion in severe_negative_emotions and target_cat == "笑话":
        return False
    # 额外检查：文案中是否出现了强烈负面词（不管情感检测如何）
    severe_hints = ["想死", "自杀", "活不下去", "不想活", "死了算", "绝望"]
    for hint in severe_hints:
        if hint in question_text.lower() and target_cat == "笑话":
            return False
    return True


def main():
    # 注意：不再设置 random.seed(42)，以确保每次运行都有真正随机的结果，
    # 特别是在第3轮索要语和第4轮故事选择上获得充分的多样性。
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
    skipped_validation = 0
    skipped_emotional_mismatch = 0  # 新增：情感适配跳过计数
    max_attempts_factor = 5  # 为防止死循环，设定最大尝试次数倍数
    max_attempts = len(counseling_data) * max_attempts_factor
    attempt = 0

    while len(fused_dialogues) < MAX_FUSED_DIALOGUES and attempt < max_attempts:
        idx = attempt % len(counseling_data)
        record = counseling_data[idx]
        attempt += 1

        # 提取情感倾诉（仅用户问题）
        question_text = extract_first_qa(record)
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

        # ---- 增强的情感适配检查 ----
        # 确保抑郁等严重负面情绪不匹配笑话
        if not _check_emotional_adaptation(emotion, target_cat, question_text):
            # 强制降级为故事
            print(f"    ⚠️ 情感适配：{emotion}情绪+'{question_text[:30]}...'强制使用故事")
            target_cat = "故事"
            skipped_emotional_mismatch += 1

        # 从对应池中随机选一条
        pool = stories_pool if target_cat == "故事" else jokes_pool
        if not pool:
            skipped_no_story += 1
            continue

        story_item = random.choice(pool)

        # 构造融合对话（不再传入 empathic_reply）
        dialogue = generate_fused_dialogue(question_text, story_item, emotion)

        # 质量验证，不合格跳过
        if not validate_fused_dialogue(dialogue):
            skipped_validation += 1
            continue

        fused_dialogues.append(dialogue)

        # 统计
        fused_count_by_emotion[emotion] = fused_count_by_emotion.get(emotion, 0) + 1

        if len(fused_dialogues) % 500 == 0:
            print(f"  已处理 {attempt} 次尝试，生成 {len(fused_dialogues)} 条融合对话...")

    # ========== 打印统计 ==========
    print_divider("融合统计")
    print(f"  {'指标':<30} {'数量':<10}")
    print(f"  {'-' * 40}")
    print(f"  {'处理咨询记录':<30} {len(counseling_data):<10,}")
    print(f"  {'跳过-无有效问题':<30} {skipped_no_question:<10,}")
    print(f"  {'跳过-无匹配故事/笑话':<30} {skipped_no_story:<10,}")
    print(f"  {'跳过-情感适配强制降级':<30} {skipped_emotional_mismatch:<10,}")
    print(f"  {'跳过-质量验证不合格':<30} {skipped_validation:<10,}")
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
