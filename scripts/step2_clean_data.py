"""
Step 2 — 脏数据清洗（考核项2-②，3%）

考核要求：
- 剔除广告引流、无效社交话术
- 文本去重清理

清洗规则：
1. 广告引流：正则匹配含广告特征的内容
2. 无效社交：单轮无意义社交话术
3. 文本去重：基于 content hash
4. 过短过滤：content < 5 字丢弃
"""
import json
import os
import re
import hashlib
import sys
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== 清洗规则 ====================

# 广告引流关键词（正则）
AD_PATTERNS = [
    r'加[Vv微信]', r'扫码', r'免费[领送]', r'点击[链接网址]',
    r'http[s]?://', r'拼单', r'优惠[券码]', r'下单',
    r'兼职[招]?[聘募]', r'公众号', r'关注.*[微v]信',
    r'qq\d{5,}', r'微信\d{5,}', r'[微v]信号',
    r'领取.*[红包福利]', r'名额.*有限', r'仅限.*[天时]',
    r'[推荐推]荐.*[产商]品', r'批[发量]',
    r'老师.*qq|qql?', r'加我.*[微vq]',
    r'免费.*[课讲]程', r'限时.*[优惠免]',
]

# 无效社交话术（完整匹配，非子串匹配，避免误伤正常内容）
INVALID_SOCIAL_PATTERNS = [
    r'^在吗[？?]?$',
    r'^有人吗[？?]?$',
    r'^睡了吗[？?]?$',
    r'^有空吗[？?]?$',
    r'^在不在[？?]?$',
    r'^你好$',
    r'^您好$',
    r'^嗯$',
    r'^哦$',
    r'^好的?$',
    r'^谢谢$',
    r'^感谢$',
    r'^谢了$',
    r'^知道[了]?$',
    r'^明白[了]?$',
    r'^是的?$',
    r'^嗯嗯$',
    r'^哈哈$',
    r'^呵呵$',
    r'^好吧$',
    r'^行吧$',
    r'^加油$',
    r'^\?\?\?$',
    r'^\.\.\.$',
    r'^。。。$',
    r'^……$',
]

# 预编译正则
AD_RE = re.compile('|'.join(AD_PATTERNS), re.IGNORECASE)
INVALID_SOCIAL_RE = re.compile('|'.join(INVALID_SOCIAL_PATTERNS))

MIN_CONTENT_LENGTH = 5


def print_divider(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def is_ad(text):
    """检测是否包含广告引流内容"""
    return bool(AD_RE.search(text))


def is_invalid_social(text):
    """检测是否为无效社交话术"""
    text = text.strip()
    return bool(INVALID_SOCIAL_RE.match(text))


def content_hash(text):
    """对文本内容计算 hash 用于去重"""
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()


def extract_all_texts(data):
    """从咨询数据中提取所有对话文本"""
    texts = []  # (source_id, text) 保留来源用于统计
    for item in data:
        qid = item.get("question_id", "")
        title = item.get("question_title", "")
        content = item.get("question_content", "")
        if title:
            texts.append((f"{qid}_title", title))
        if content:
            texts.append((f"{qid}_content", content))

        for answer in item.get("answers", []):
            for dialog in answer.get("dialogs", []):
                dialog_text = dialog.get("content", "")
                if dialog_text:
                    texts.append((f"{qid}_dialog", dialog_text))
    return texts


def clean_data(data):
    """执行完整清洗流程，返回清洗后数据 + 统计信息"""
    stats = {
        "total_original": 0,
        "removed_ad": 0,
        "removed_social": 0,
        "removed_short": 0,
        "removed_duplicate": 0,
        "kept": 0,
    }

    # 第一步：提取所有文本并逐条判断
    all_texts = extract_all_texts(data)
    stats["total_original"] = len(all_texts)

    # 逐条清洗
    cleaned_texts = []
    seen_hashes = set()
    removed_ad_msgs = []
    removed_social_msgs = []

    for src_id, text in all_texts:
        # 1. 广告过滤
        if is_ad(text):
            stats["removed_ad"] += 1
            removed_ad_msgs.append(text[:80])
            continue

        # 2. 无效社交过滤
        if is_invalid_social(text):
            stats["removed_social"] += 1
            removed_social_msgs.append(text[:80])
            continue

        # 3. 过短过滤
        if len(text.strip()) < MIN_CONTENT_LENGTH:
            stats["removed_short"] += 1
            continue

        # 4. 去重
        h = content_hash(text)
        if h in seen_hashes:
            stats["removed_duplicate"] += 1
            continue
        seen_hashes.add(h)

        cleaned_texts.append((src_id, text))
        stats["kept"] += 1

    return cleaned_texts, stats, removed_ad_msgs[:10], removed_social_msgs[:10]


def count_ad_removal(data):
    """统计完整记录级广告（用于报表）"""
    total = len(data)
    ad_in_title = 0
    ad_in_content = 0
    for item in data:
        title = item.get("question_title", "")
        content = item.get("question_content", "")
        if is_ad(title):
            ad_in_title += 1
        if is_ad(content):
            ad_in_content += 1
    return total, ad_in_title, ad_in_content


def print_stats(stats, label=""):
    """打印清洗统计表"""
    if label:
        print(f"\n  >>> {label}")
    print(f"  {'指标':<30} {'数量':<10} {'占比':<10}")
    print(f"  {'-' * 50}")
    print(f"  {'原始文本总条数':<30} {stats['total_original']:<10,} {'100%':<10}")
    print(f"  {'移除-广告引流':<30} {stats['removed_ad']:<10,} {stats['removed_ad'] / max(stats['total_original'], 1) * 100:.1f}%")
    print(f"  {'移除-无效社交':<30} {stats['removed_social']:<10,} {stats['removed_social'] / max(stats['total_original'], 1) * 100:.1f}%")
    print(f"  {'移除-过短(<5字)':<30} {stats['removed_short']:<10,} {stats['removed_short'] / max(stats['total_original'], 1) * 100:.1f}%")
    print(f"  {'移除-重复':<30} {stats['removed_duplicate']:<10,} {stats['removed_duplicate'] / max(stats['total_original'], 1) * 100:.1f}%")
    print(f"  {'-' * 50}")
    print(f"  {'清洗后保留':<30} {stats['kept']:<10,} {stats['kept'] / max(stats['total_original'], 1) * 100:.1f}%")


def merge_stats(*stats_list):
    """合并多个统计字典"""
    merged = {k: 0 for k in stats_list[0]} if stats_list else {}
    for s in stats_list:
        for k in merged:
            merged[k] += s.get(k, 0)
    return merged


def clean_record_level(item):
    """对单条记录执行对话级清洗，返回清洗后的记录或 None"""
    title = item.get("question_title", "")
    content = item.get("question_content", "")

    # 标题和正文都是广告 → 整条记录丢弃
    if title and is_ad(title) and content and is_ad(content):
        return None

    new_answers = []
    for answer in item.get("answers", []):
        new_dialogs = []
        for dialog in answer.get("dialogs", []):
            text = dialog.get("content", "")
            if text and not is_ad(text) and not is_invalid_social(text) and len(text.strip()) >= MIN_CONTENT_LENGTH:
                new_dialogs.append(dialog)

        if new_dialogs:
            new_answer = dict(answer)
            new_answer["dialogs"] = new_dialogs
            new_answers.append(new_answer)

    if not new_answers:
        return None

    new_item = dict(item)
    new_item["answers"] = new_answers
    return new_item


def main():
    print("=" * 60)
    print("  Step 2 — 脏数据清洗")
    print("  考核项(2)-② 3%：剔除广告引流、无效社交话术、去重")
    print("  ⚡ 覆盖两份数据集：jiandanxinli + psy525")
    print("=" * 60)

    # ==================== 数据集1：jiandanxinli 情感问答 ====================
    print_divider("数据集1：jiandanxinli_qa 情感问答")
    fpath1 = os.path.join(BASE_DIR, "jiandanxinli_qa_data_v1.0.json")
    with open(fpath1, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    print(f"  已加载：{len(data1)} 条原始记录")

    total, ad_title, ad_content = count_ad_removal(data1)
    print(f"  含广告特征标题：{ad_title}条 | 含广告特征正文：{ad_content}条")

    c1_texts, stats1, ad_samples1, social_samples1 = clean_data(data1)
    print_stats(stats1, "jiandanxinli 清洗统计")

    # ==================== 数据集2：psy525 补充数据 ====================
    print_divider("数据集2：psy525 补充数据 (data/No-*.json)")
    data_dir = os.path.join(BASE_DIR, "data")
    psy525_cleaned_texts = []  # 所有 psy525 的清洗后文本
    psy525_all_records = []    # 所有 psy525 原始记录
    stats_list = []            # 各文件清洗统计列表
    ad_samples_psy = []
    social_samples_psy = []
    psy525_file_count = 0
    psy525_loaded = 0

    # 查找 data/No-*.json 文件
    no_files = sorted([f for f in os.listdir(data_dir) if re.match(r'^No-\d+\.json$', f)],
                      key=lambda x: int(re.search(r'\d+', x).group()))
    if not no_files:
        print("  ⚠ 未找到 data/No-*.json 文件，跳过 psy525 数据集")
    else:
        print(f"  发现 {len(no_files)} 个 No-*.json 文件")

        for fname in no_files:
            fpath = os.path.join(data_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
            except Exception as e:
                print(f"  ⚠ 跳过 {fname}：读取失败 ({e})")
                continue

            if not isinstance(file_data, list) or len(file_data) == 0:
                print(f"  ⚠ 跳过 {fname}：数据为空或非列表格式")
                continue

            psy525_file_count += 1
            psy525_loaded += len(file_data)

            # 执行清洗
            texts, st, ad_s, soc_s = clean_data(file_data)

            psy525_cleaned_texts.extend(texts)
            psy525_all_records.extend(file_data)
            stats_list.append(st)
            ad_samples_psy.extend(ad_s)
            social_samples_psy.extend(soc_s)

            if psy525_file_count <= 3:
                print(f"    {fname}: {len(file_data)} 条 → 保留 {st['kept']} 条文本")

        if psy525_file_count > 3:
            print(f"    ... 共处理 {psy525_file_count} 个文件 ({psy525_loaded:,} 条原始记录)")

    # 合并 psy525 统计
    if stats_list:
        stats2 = merge_stats(*stats_list)
        print_stats(stats2, "psy525 清洗统计（汇总）")
    else:
        stats2 = None
        print("  psy525 数据集无数据，跳过")

    # ========== 打印被删样本 ==========
    all_ad_samples = ad_samples1 + ad_samples_psy[:10]
    all_social_samples = social_samples1 + social_samples_psy[:10]

    if all_ad_samples:
        print_divider("广告引流样本（前10条）")
        for s in all_ad_samples[:10]:
            print(f"  [AD] {s}")

    if all_social_samples:
        print_divider("无效社交样本（前10条）")
        for s in all_social_samples[:10]:
            print(f"  [SOCIAL] {s}")

    # ========== 合并全局统计 ==========
    print_divider("两份数据集合并统计")
    if stats2:
        all_texts_combined = c1_texts + psy525_cleaned_texts
        global_stats = merge_stats(stats1, stats2)
        global_stats["kept"] = len(all_texts_combined)
    else:
        all_texts_combined = c1_texts
        global_stats = stats1

    print_stats(global_stats, "全局汇总")

    # ========== 保存清洗后数据 ==========
    output_path = os.path.join(BASE_DIR, "data", "cleaned", "cleaned_data.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 构建保留的清洗后记录（保持原始结构）
    cleaned_records = []
    for item in data1:
        cleaned = clean_record_level(item)
        if cleaned is not None:
            cleaned_records.append(cleaned)

    if psy525_all_records:
        for item in psy525_all_records:
            cleaned = clean_record_level(item)
            if cleaned is not None:
                cleaned_records.append(cleaned)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_records, f, ensure_ascii=False, indent=2)

    print(f"\n  已保存清洗后数据：{output_path}")
    print(f"  数据源：jiandanxinli {len(data1)} 条 + psy525 {psy525_loaded} 条 = 合计 {psy525_loaded + len(data1)} 条记录")
    print(f"  清洗后保留：{len(cleaned_records)} 条记录 ({len(all_texts_combined)} 条文本)")
    print(f"\n✅ Step 2 完成。建议截图：以上两份清洗统计报告及合并统计")


if __name__ == "__main__":
    main()
