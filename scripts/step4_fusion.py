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

# 融合对话数上限（大幅提升以覆盖多种对话模式）
MAX_FUSED_DIALOGUES = 8000

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


# ====== 新增：纯共情模板（不推荐故事/笑话，只做心理咨询） ======
EMPATHY_NO_RECOMMEND = {
    "焦虑": [
        "我能感受到你的焦虑，这种感觉真的很不好受。你愿意多说说是哪些事情让你这么紧张吗？",
        "焦虑的时候有人倾听就很重要。我在，你慢慢说。",
        "试试深呼吸，把注意力放在当下。你能描述一下现在最让你担心的是什么吗？",
        "压力确实会让人透不过气。你平时有什么方式可以让自己稍微放松一点吗？",
    ],
    "抑郁": [
        "我听到你的难过了，这种感受很真实。你愿意跟我多聊聊吗？",
        "你愿意把这些说出来已经很勇敢了。这种感觉持续多久了？",
        "低落的时候，也许不需要急着走出来，我们可以先聊聊。",
        "我在这里陪着你。你想说说最近发生了什么事吗？",
    ],
    "人际": [
        "和别人相处确实有时候会让人觉得很累。你想聊聊具体是什么情况吗？",
        "孤独的感觉我懂。你是希望交到更多朋友，还是对现有的关系有些困扰？",
        "人际关系常常是最消耗心力的。能和我说说让你不舒服的是什么吗？",
    ],
    "家庭": [
        "家人之间的冲突真的特别让人难受。你想多说说发生了什么吗？",
        "和家人闹矛盾的时候心里一定很委屈，我在这儿听你说。",
        "家本该是最温暖的地方，有时候却让人最受伤。你愿意多聊聊吗？",
    ],
    "职场": [
        "工作压力确实让人喘不过气。你愿意具体说说是什么情况吗？",
        "每天都在为了工作奔波，真的很不容易。你最近遇到什么困难了？",
        "职场里的很多事都身不由己，我理解你的感受。",
    ],
    "学业": [
        "学业上的压力确实很磨人。你最近在备考什么？",
        "学习这条路有时候真的会让人很迷茫，你觉得自己现在最大的困难是什么？",
        "成绩起伏很正常，重要的是你没有放弃。你愿意多聊聊吗？",
    ],
    "日常": [
        "谢谢你愿意跟我分享这些。还有什么想聊聊的吗？",
        "我在这儿听你说，你尽管说吧。",
        "每种情绪都值得被认真对待，你想继续聊聊吗？",
    ],
}

# ====== 新增：用户继续倾诉语（替代索要故事/笑话） ======
USER_CONTINUE_PHRASES = [
    "我也不知道该怎么办，就是很难受",
    "我试过很多方法都没用",
    "有时候我觉得是不是我自己的问题",
    "周围没人理解我，我只能自己扛着",
    "你能理解这种感觉吗？",
    "谢谢你听我说这些",
    "这种情况已经持续很久了",
    "我真的很迷茫",
    "说不清楚，就是心里堵得慌",
    "感觉活着好累啊",
    "我不知道跟谁说，只能跟你说",
    "其实不只是这件事，还有好多事让我烦",
    "嗯...我也不知道从哪里说起",
    "你说我是不是太矫情了",
    "其实我平时不是这样的，但最近...",
    "家里人也不理解我，觉得我想太多",
    "我试过跟朋友说，但他们都不当回事",
    "有时候早上醒来就不想起床",
    "我不知道这种状态还要持续多久",
    "谢谢你，终于有人愿意认真听我说话了",
]

# ====== 新增：用户拒绝推荐语 ======
USER_DECLINE_PHRASES = [
    "不用了，我就想聊聊",
    "不想听故事，你能理解我的感受吗？",
    "算了，听了也开心不起来",
    "其实我更想知道怎么解决这个问题",
    "不用讲笑话了，你能给我一些建议吗？",
    "我现在笑不出来...",
    "听了也没用，我的问题不是故事能解决的",
    "不要了吧，我就想有个人听我说说话",
    "其实我不太想听这些，你陪我聊聊就好",
    "你知道吗，这些笑话和故事对我没用的",
    "别讲那些了，我真正需要的是有人理解我",
    "说实话我现在什么都不想听，就想哭一场",
]

# ====== 新增：被拒绝后的尊重回复 ======
FALLBACK_EMPATHY = {
    "焦虑": [
        "没关系，不想听故事也没事。说说看，现在最让你焦虑的是什么呢？",
        "好的，不勉强你。我们聊聊你最近的具体情况吧。",
        "理解，那我们就这样聊聊。你觉得焦虑的来源主要是什么？",
    ],
    "抑郁": [
        "理解，我现在就陪着你。你想聊什么都可以。",
        "没关系的，我们就这样聊聊也挺好。你最近睡眠怎么样？",
        "好的，你说得对，有些感受不是故事能解决的。你愿意多说说吗？",
    ],
    "人际": [
        "好的，那我就在这儿听你说。你觉得人际关系中最困扰你的是什么？",
        "没关系，我们继续聊聊你的感受吧。",
    ],
    "家庭": [
        "好的，那我们继续聊聊。你和家人的关系是从什么时候开始变紧张的？",
        "没关系，家庭问题确实不是简单一个故事能解决的。你多跟我说说。",
    ],
    "职场": [
        "理解，工作上的压力确实不是笑话就能化解的。你现在的工作状态怎么样？",
        "好的，我们聊聊你的情况。你觉得最让你累的是什么？",
    ],
    "学业": [
        "好的，那我们继续聊聊。你现在的学习状态怎么样？",
        "没关系，考试压力确实不是笑话能解决的。你备考遇到什么困难了？",
    ],
    "日常": [
        "没关系，我们就这样聊聊也挺好。",
        "好的，你想聊什么都可以，我在这儿听着。",
    ],
}

# ====== 新增：深度共情追问（纯共情模式第4轮） ======
FOLLOWUP_EMPATHY = {
    "焦虑": [
        "这种感觉一定很辛苦。焦虑往往是因为我们在乎，但这份在乎也变成了负担。你觉得除了聊天，还有什么能让你稍微好受一点？",
        "我理解。焦虑就像是心里有一只停不下来的陀螺。你能试着说说最让你安心的时刻是什么吗？",
    ],
    "抑郁": [
        "谢谢你的信任。低落的时候，未来好像都蒙上了一层灰。但我想让你知道，你并不孤单。",
        "你辛苦了。有时候情绪就是会莫名其妙地低落，这不是你的错。你想聊聊那些让你稍微感到温暖的小事吗？",
    ],
    "人际": [
        "每个人都是一座孤岛，但人与人之间的桥梁是需要时间去搭建的。你不必急于改变自己。",
        "能感受到你的孤独。和人交往有时候就像学一门语言，需要不断尝试和练习。",
    ],
    "家庭": [
        "家是我们最深的牵绊也是最容易受伤的地方。你愿意聊聊你期望的家是什么样的吗？",
        "家庭关系真的很复杂，爱中掺杂着期待和失望。但无论如何，你的感受都是重要的。",
    ],
    "职场": [
        "职场的压力有时候真的让人喘不过气。除了工作，你还有哪些能让你感到快乐的事情？",
        "能理解你的疲惫。有时候我们需要给自己的生活留一点缝隙，让光透进来。",
    ],
    "学业": [
        "学习是一场长跑不是短跑，累了就歇一歇。你已经很努力了，这不是空话。",
        "我理解那种付出了却看不到结果的感觉。但每一个脚步都在积累，虽然现在可能看不到。",
    ],
    "日常": [
        "谢谢你跟我说这些。生活中的每一个小情绪都值得被温柔对待。",
        "每个人都需要一个倾诉的出口，很高兴你愿意让我成为这个出口。",
    ],
}

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


def _gen_empathic_no_recommend(emotion):
    """纯共情回复：不推荐故事/笑话，只做心理咨询"""
    templates = EMPATHY_NO_RECOMMEND.get(emotion, EMPATHY_NO_RECOMMEND["日常"])
    return random.choice(templates)


def _gen_followup_empathy(emotion):
    """深度共情追问（纯共情模式第4轮）"""
    templates = FOLLOWUP_EMPATHY.get(emotion, FOLLOWUP_EMPATHY["日常"])
    return random.choice(templates)


def _gen_fallback_empathy(emotion):
    """被用户拒绝后的尊重回复"""
    templates = FALLBACK_EMPATHY.get(emotion, FALLBACK_EMPATHY["日常"])
    return random.choice(templates)


# 第5轮用户切换话题语
USER_SWITCH_TOPIC_PHRASES = [
    "其实我还想说说另一件事...",
    "不说这个了，我还有个问题想问你",
    "对了，我最近还遇到了一个情况",
    "说到这个，我还有其他方面的困扰",
    "这件事先放一边，还有件事让我更难受",
]


def generate_fused_dialogue(question_text, story_item, emotion):
    """构造多样化的多轮对话

    5种对话模式，覆盖真实咨询中的各种情况：

      - recommend (25%):    四轮推荐模式 — 共情→推荐→索要→讲故事
      - pure_empathy (25%): 四轮纯共情 — 共情→倾诉→深度回应（不推荐）
      - user_decline (15%): 四轮拒绝 — 共情→推荐→拒绝→尊重
      - short_empathy (15%): 二轮短对话 — 倾诉→共情（不推荐，不索要）
      - extended (20%):     五轮长对话 — 情感倾诉→推荐→索要→讲故事→追问→回应

    这样模型学到的不是"死板的推荐流程"，而是"根据不同情境灵活应对"。
    """
    story_text = story_item.get("text", "") or story_item.get("content", "")
    category = story_item.get("category", "其他")

    pattern = random.choices(
        ["recommend", "pure_empathy", "user_decline", "short_empathy", "extended"],
        weights=[0.25, 0.25, 0.15, 0.15, 0.20],
        k=1,
    )[0]

    # ---- 模式1：推荐（原有逻辑）----
    if pattern == "recommend":
        empathic_reply = generate_empathic_reply(emotion)
        if category == "笑话":
            ask_reason = "笑一笑心情会好一些"
        else:
            ask_reason = "温暖的故事最能治愈人心了"
        user_ask = _next_user_ask()

        dialogue = {
            "emotion": emotion, "category": category,
            "conversations": [
                {"role": "user", "content": question_text},
                {"role": "assistant", "content": f"{empathic_reply} 要听个{category}吗？{ask_reason}"},
                {"role": "user", "content": user_ask},
                {"role": "assistant", "content": story_text},
            ]
        }

    # ---- 模式2：纯共情（不推荐故事/笑话）----
    elif pattern == "pure_empathy":
        reply1 = _gen_empathic_no_recommend(emotion)
        user_continue = random.choice(USER_CONTINUE_PHRASES)
        reply2 = _gen_followup_empathy(emotion)

        dialogue = {
            "emotion": emotion, "category": "无",
            "conversations": [
                {"role": "user", "content": question_text},
                {"role": "assistant", "content": reply1},
                {"role": "user", "content": user_continue},
                {"role": "assistant", "content": reply2},
            ]
        }

    # ---- 模式3：用户拒绝推荐 ----
    elif pattern == "user_decline":
        empathic_reply = generate_empathic_reply(emotion)
        if category == "笑话":
            ask_reason = "笑一笑心情会好一些"
        else:
            ask_reason = "温暖的故事最能治愈人心了"
        user_decline = random.choice(USER_DECLINE_PHRASES)
        fallback = _gen_fallback_empathy(emotion)

        dialogue = {
            "emotion": emotion, "category": category,
            "conversations": [
                {"role": "user", "content": question_text},
                {"role": "assistant", "content": f"{empathic_reply} 要听个{category}吗？{ask_reason}"},
                {"role": "user", "content": user_decline},
                {"role": "assistant", "content": fallback},
            ]
        }

    # ---- 模式4：短对话（纯共情2轮，不推荐不索要）----
    elif pattern == "short_empathy":
        reply1 = _gen_empathic_no_recommend(emotion)

        dialogue = {
            "emotion": emotion, "category": "无",
            "conversations": [
                {"role": "user", "content": question_text},
                {"role": "assistant", "content": reply1},
            ]
        }

    # ---- 模式5：五轮长对话（推荐→索要→讲故事→追问→回应）----
    else:
        empathic_reply = generate_empathic_reply(emotion)
        if category == "笑话":
            ask_reason = "笑一笑心情会好一些"
        else:
            ask_reason = "温暖的故事最能治愈人心了"
        user_ask = _next_user_ask()
        user_followup = random.choice(USER_SWITCH_TOPIC_PHRASES)
        followup_reply = _gen_empathic_no_recommend(emotion)

        dialogue = {
            "emotion": emotion, "category": category,
            "conversations": [
                {"role": "user", "content": question_text},
                {"role": "assistant", "content": f"{empathic_reply} 要听个{category}吗？{ask_reason}"},
                {"role": "user", "content": user_ask},
                {"role": "assistant", "content": story_text[:200] + "..." if len(story_text) > 200 else story_text},
                {"role": "user", "content": user_followup},
                {"role": "assistant", "content": followup_reply},
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
    """检查融合对话的基本质量，不合格返回False

    改进：适应多种对话模式，不再强制第2轮必须推荐故事/笑话。
    - recommend 模式：四轮，第2轮含推荐
    - pure_empathy 模式：四轮，第2轮不含推荐（纯共情）
    - user_decline 模式：四轮，第2轮推荐 + 第3轮拒绝
    """
    conversations = dialogue.get("conversations", [])
    num_turns = len(conversations)
    # 支持 2/4/5 轮对话结构
    if num_turns not in (2, 4, 5):
        return False

    # 检查是否有空内容
    for turn in conversations:
        content = turn.get("content", "").strip()
        if not content:
            return False

    # 第2轮及之后内容由 generate_fused_dialogue() 担保非空且语义合理

    # ---- 内容安全过滤：仅对含故事/笑话的轮次检查敏感词 ----
    # 安全过滤的检查轮次：故事内容在 4/5 轮对话中的最后一轮assistant（索引 3）
    # 2 轮对话不含故事内容，跳过敏感词检查
    if num_turns >= 4:
        story_round = conversations[3]  # 4轮对话: 第4轮; 5轮对话: 第4轮是故事
        story_content = story_round.get("content", "")
        story_lower = story_content.lower()

        # 只有实际含故事内容时才做敏感词检查
        if dialogue.get("category") not in ("无", None):
            for kw in BLOCKED_KEYWORDS:
                if kw in story_content or kw in story_lower:
                    print(f"    ⛔ 跳过含敏感词[{kw}]的融合对话")
                    return False

        # ---- 检查第1轮和故事轮组合的安全性 ----
        round1_content = conversations[0].get("content", "")
        round1_lower = round1_content.lower()
        severe_negative_keywords = ["想死", "自杀", "活不下去", "不想活", "死了算"]
        for severe_kw in severe_negative_keywords:
            if severe_kw in round1_lower:
                if dialogue.get("category") not in ("无", "故事", None):
                    print(f"    ⛔ 用户'{severe_kw}'但推荐了笑话，跳过（必须用暖心故事）")
                    return False
                for kw in BLOCKED_KEYWORDS:
                    if kw in story_content or kw in story_lower:
                        print(f"    ⛔ 严重用户场景下第4轮含敏感词[{kw}]，跳过")
                        return False
                break

        # 故事/笑话正文长度检查
        if dialogue.get("category") not in ("无", None) and len(story_content) < 30:
            return False

        # 空洞内容检查
        stripped = story_content.strip()
        import re
        cleaned = re.sub(r'[\s,，。！？、；：""''【】《》（）\!\?\.,;:\(\)\[\]\{\}]', '', stripped)
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
