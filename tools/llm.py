# 前置依赖说明：本代码复用 OpenAI 官方 SDK 兼容调用 DeepSeek 大模型接口
# DeepSeek 接口完全兼容 OpenAI 调用格式，因此直接使用 openai 库即可对接
# 安装依赖命令：pip3 install openai
import os
# 从openai库导入OpenAI客户端类，用于构建API请求客户端
from openai import OpenAI

# ===================== 初始化DeepSeek客户端 =====================
client = OpenAI(
    # 读取系统环境变量中存储的DeepSeek密钥DEEPSEEK_API_KEY
    # 不建议直接把密钥写在代码里，存在泄露风险，统一通过环境变量读取更安全
    #api_key=os.environ.get('DEEPSEEK_API_KEY'),
    #api_key="sk-cd6468846e7a4570b8bca4580fa9e64f",
    api_key="ark-d25f9cd7-14a5-41f9-a31c-f9f21f43eac4-8ab2f",
    # 指定DeepSeek官方API请求地址，覆盖OpenAI默认接口地址
    #base_url="https://api.deepseek.com"
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

# ==================== 系统提示词 ====================
SYSTEM_PROMPT = """你是NLP专家，只输出与NLP相关的内容，简洁。
1. 输出文本的分类：故事、笑话、诗歌、散文、新闻、评论、对话、其他
2. 提取文本中最相关的的关键词，2个
"""
# ===================== 发起对话补全请求 =====================
response = client.chat.completions.create(
    # 指定要调用的DeepSeek模型名称，deepseek-v4-pro为深度求索高端推理模型
    #model="deepseek-v4-pro",
    model="doubao-seed-2-0-pro-260215",
    # 对话上下文消息列表，遵循OpenAI标准消息结构
    messages=[
        # system角色：系统提示词，设定AI助手身份、行为规范、回答风格
        {"role": "system", "content": SYSTEM_PROMPT },
        # user角色：用户输入的提问内容，本轮用户输入为Hello
        {"role": "user", "content": "在饭店里。 一名旅客问：“服务员，把你们的电话号码簿拿给我，我要找个 地址。” “很抱歉，先生，我们这里没有电话号码簿，不过我倒是可以把 意见簿拿给您，您可以从上面找到我们这个城市几乎所有的居民的地址。”"},
    ],
    # stream=False：关闭流式输出，一次性返回完整回答
    # 若设为True逐块返回文字，适合实现打字机实时输出效果
    stream=False,
    # reasoning_effort="high"：DeepSeek专属参数，控制模型推理思考强度
    # high=高推理力度，深层逻辑推演，适配数学、代码、复杂逻辑问题
    # 可选值：low / medium / high，推理强度越高耗时越长
    reasoning_effort="high",
    # extra_body：传递OpenAI标准未定义的厂商自定义扩展参数
    # {"thinking": {"type": "enabled"}}：开启DeepSeek思考过程输出
    # 开启后响应会附带模型内部推理步骤，可单独提取查看
    extra_body={"thinking": {"type": "enabled"}}
)

# ===================== 解析并打印AI返回结果 =====================
# response：接口返回的完整响应对象
# .choices[0]：取第一条AI回复（单次请求默认生成1条回答，索引0）
# .message.content：提取AI最终输出的回答文本内容
print(response.choices[0].message.content)

# ===================== 补充扩展知识点 =====================
"""
1. 环境变量设置方法（不同操作系统）
Windows cmd：set DEEPSEEK_API_KEY=你的密钥
Windows PowerShell：$env:DEEPSEEK_API_KEY="你的密钥"
Mac / Linux：export DEEPSEEK_API_KEY="你的密钥"

2. 提取并打印模型思考过程代码
# 判断返回对象是否存在思考内容字段
if hasattr(response.choices[0].message, 'thinking_content'):
    print("模型推理思考过程：", response.choices[0].message.thinking_content)

3. 参数补充说明
- reasoning_effort 仅DeepSeek系列模型支持，原生OpenAI无此参数
- stream=True 适用场景：网页聊天实时打字、长文本分段展示
- extra_body 为厂商扩展字段，仅DeepSeek、通义千问等兼容接口可用
"""