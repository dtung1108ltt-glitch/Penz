from setuptools import setup, find_packages

setup(
    name="voice_ai_system",
    version="1.0.0",
    description="Production-grade End-to-End Voice AI Assistant System",
    author="Voice AI Team",
    packages=find_packages(include=["config*", "core*", "engines*", "pipeline*", "scripts*", "training*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "sounddevice>=0.4.6",
        "pydantic>=2.5.0",
        "python-dotenv>=1.0.0",
        "rich>=13.7.0",
    ],
    extras_require={
        "inference": [
            "faster-whisper>=1.0.0",
            "edge-tts>=6.1.12",
            "ollama>=0.2.0",
            "openai>=1.12.0",
        ],
        "training": [
            "torch>=2.1.0",
            "torchaudio>=2.1.0",
            "transformers>=4.38.0",
            "peft>=0.9.0",
            "datasets>=2.18.0",
            "accelerate>=0.27.0",
            "jiwer>=3.0.3",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "voice-assistant=main:main",
        ],
    },
)
