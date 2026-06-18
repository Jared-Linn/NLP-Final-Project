"""
自然语言处理期末项目 - setup.py
"""
from setuptools import setup, find_packages

setup(
    name="nlp-dialogue-system",
    version="1.0.0",
    description="自然语言处理期末项目：基于 Qwen3.5 的多轮对话系统",
    author="NLP Final Project",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "peft>=0.7.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "jinja2>=3.1.0",
        "openai>=1.6.0",
    ],
    include_package_data=True,
)
