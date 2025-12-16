
1. 基础示例 - 图片 URL
import requests

url = "https://api.apiyi.com/v1/chat/completions"
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

payload = {
    "model": "gpt-4o",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "请详细描述这张图片的内容"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg"
                    }
                }
            ]
        }
    ]
}

response = requests.post(url, headers=headers, json=payload)
result = response.json()
print(result['choices'][0]['message']['content'])
​
2. 本地图片示例 - Base64 编码
import base64
import requests

def image_to_base64(image_path):
    """将本地图片转换为 base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 读取本地图片
base64_image = image_to_base64("path/to/your/image.jpg")

url = "https://api.apiyi.com/v1/chat/completions"
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

payload = {
    "model": "gemini-2.5-pro",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "分析这张图片中的所有文字内容"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
}

response = requests.post(url, headers=headers, json=payload)
print(response.json()['choices'][0]['message']['content'])
​
3. 高级示例 - 多图对比分析
import requests

url = "https://api.apiyi.com/v1/chat/completions"
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

payload = {
    "model": "gpt-4o",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请对比这两张图片的差异："},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image1.jpg"}
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": "https://example.com/image2.jpg"}
                }
            ]
        }
    ],
    "max_tokens": 1000
}

response = requests.post(url, headers=headers, json=payload)
print(response.json()['choices'][0]['message']['content'])
​
🎯 常见应用场景
​
1. 商品识别与分析
prompt = """
请分析这张商品图片，包括：
1. 商品类型和品牌
2. 主要特征和卖点
3. 适合的目标用户
4. 建议的营销文案
"""
​
2. 文档 OCR 识别
prompt = """
请提取图片中的所有文字内容，并按照原始格式整理输出。
如果有表格，请用 Markdown 表格格式呈现。
"""
​
3. 医学影像辅助分析
prompt = """
这是一张医学影像图片，请：
1. 描述图像的基本信息（如成像类型、部位等）
2. 标注可见的解剖结构
3. 注意：仅供参考，不作为诊断依据
"""
​
4. 安全监控场景分析
prompt = """
分析监控画面，识别：
1. 场景中的人数和位置
2. 是否有异常行为
3. 环境安全隐患
4. 时间戳信息（如果可见）
"""
​
💡 最佳实践
​
图片预处理建议
格式支持：JPEG、PNG、GIF、WebP 等主流格式
大小限制：建议单张图片不超过 20MB
分辨率：高分辨率图片会获得更好的识别效果
压缩优化：适度压缩以提高传输速度
​
提示词优化
# ❌ 不推荐：模糊的提示
prompt = "看看这是什么"

# ✅ 推荐：具体明确的提示
prompt = """
请从以下几个方面分析这张图片：
1. 主要对象：识别图片中的主要物体或人物
2. 场景环境：描述拍摄地点和环境特征
3. 色彩构图：分析配色方案和构图特点
4. 情感氛围：图片传达的情绪或氛围
5. 可能用途：这张图片适合用于什么场景
"""
​
错误处理
import requests
from requests.exceptions import RequestException

def analyze_image_with_retry(image_url, prompt, max_retries=3):
    """带重试机制的图像分析函数"""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.apiyi.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print(f"速率限制，等待后重试... (尝试 {attempt + 1}/{max_retries})")
                time.sleep(2 ** attempt)  # 指数退避
            else:
                print(f"错误: {response.status_code} - {response.text}")
                
        except RequestException as e:
            print(f"请求异常: {e}")
            
    return None
​
🔧 高级功能
​
1. 流式输出
对于长篇分析，可以使用流式输出获得更好的用户体验：
payload = {
    "model": "gpt-4o",
    "messages": [...],
    "stream": True
}

response = requests.post(url, headers=headers, json=payload, stream=True)
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
​
2. 多轮对话
保持上下文进行深入分析：
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "这是什么动物？"},
            {"type": "image_url", "image_url": {"url": "animal.jpg"}}
        ]
    },
    {
        "role": "assistant",
        "content": "这是一只金毛寻回犬。"
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "它看起来多大了？健康状况如何？"}]
    }
]
​
3. 结合函数调用
tools = [
    {
        "type": "function",
        "function": {
            "name": "save_image_analysis",
            "description": "保存图像分析结果到数据库",
            "parameters": {
                "type": "object",
                "properties": {
                    "objects": {"type": "array", "items": {"type": "string"}},
                    "scene": {"type": "string"},
                    "text_content": {"type": "string"}
                }
            }
        }
    }
]

payload = {
    "model": "gpt-4o",
    "messages": messages,
    "tools": tools,
    "tool_choice": "auto"
}
