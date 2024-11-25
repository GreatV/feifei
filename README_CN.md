# 肥肥：GitHub自动化 Issue 回复机器人

[English](README.md) | [中文](README_CN.md)

该仓库包含一个Python脚本，使用语言模型（LLM）自动回复指定GitHub仓库中的新Issue。该机器人利用OpenAI的GPT模型、Hugging Face模型或其他支持的模型，根据仓库内容生成回复。

#### 特性

- **自动回复**：监控指定的GitHub仓库中的新Issue，并使用LLM自动生成回复。
- **LLM集成**：支持OpenAI、Hugging Face、Anthropic、Google、Qianfan、Ollama或本地模型来生成回复。
- **向量存储**：构建或加载仓库内容的向量存储，实现基于上下文的回复。
- **环境配置**：使用环境变量和YAML配置文件，灵活设置。
- **线程安全操作**：实现锁机制，确保文件操作的线程安全。
- **可扩展设计**：模块化函数，方便定制和扩展功能。

#### 目录

- [先决条件](#先决条件)
- [安装](#安装)
- [配置](#配置)
- [使用方法](#使用方法)
- [自定义](#自定义)
- [贡献](#贡献)
- [许可证](#许可证)

#### 先决条件

- Python 3.10或更高版本
- GitHub、OpenAI、Hugging Face或其他支持的后端的访问令牌

#### 安装

1. **克隆仓库**

   ```bash
   https://github.com/GreatV/feifei.git
   cd feifei
   ```

2. **创建虚拟环境**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # 在Windows上使用 'venv\Scripts\activate'
   ```

3. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

#### 配置

##### 环境变量

在根目录创建一个 `.env` 文件，添加以下变量：

```env
GITHUB_TOKEN=your_github_token
OPENAI_API_KEY=your_openai_api_key  # 可选，如果使用OpenAI
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token  # 可选，如果使用Hugging Face
```

##### YAML配置

在根目录创建或修改 `config.yaml` 文件：

```yaml
github_base_url: "https://api.github.com"
llm_backend: "openai"  # 选项：'openai'，'huggingface'，'local'
embedding_backend: "openai"  # 选项：'openai'，'huggingface'，'local'
model_name: "gpt-3.5-turbo"  # LLM模型名称
embedding_model_name: "sentence-transformers/all-MiniLM-L6-v2"
check_interval: 300  # 检查间隔时间（秒）
repositories:
  yourusername/your-repo:
    start_issue_number: 1
```

- **github_base_url**：GitHub API请求的基本URL。
- **llm_backend**：使用的语言模型后端。
- **embedding_backend**：使用的嵌入模型后端。
- **model_name**：LLM模型的名称。
- **embedding_model_name**：嵌入模型的名称。
- **check_interval**：机器人检查新Issue的间隔时间（秒）。
- **repositories**：要监控的仓库列表及其设置。

#### 使用方法

运行主脚本：

```bash
python main.py
```

您可以使用 `--debug` 命令行参数来启用调试模式：

```bash
python main.py --debug
```

机器人将开始监控指定的仓库中的新Issue并自动回复。

#### 自定义

##### 更改LLM后端

您可以通过更新 `config.yaml` 中的 `llm_backend` 来在OpenAI、Hugging Face或本地模型之间切换：

```yaml
llm_backend: "huggingface"  # 或 'local'
```

如果选择 'local'，请确保您已安装必要的令牌和模型。

##### 修改提示模板

可以在 `get_prompt_template()` 函数中自定义生成回复所使用的提示：

```python
def get_prompt_template():
    prompt_template = """
You are a smart assistant capable of answering user questions based on provided documents.

Question: {input}

Here are some potentially useful documents:
{context}

Please provide a detailed answer based on the above information, in the same language as the question. If possible, reference related Issues or Discussions and provide links.

Answer:
"""
    return PromptTemplate(
        template=prompt_template, input_variables=["input", "context"]
    )
```

##### 调整检索设置

可以在 `search_kwargs` 参数中调整用于提供上下文的检索文档数量：

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

#### 贡献

欢迎贡献！请按照以下步骤：

1. **Fork仓库**

2. **创建新分支**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **提交更改**

   ```bash
   git commit -am 'Add new feature'
   ```

4. **推送到分支**

   ```bash
   git push origin feature/your-feature-name
   ```

5. **创建Pull Request**

#### 许可证

本项目采用 [Apache-2.0 许可证](LICENSE)。
