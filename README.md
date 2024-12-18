# FeiFei: Automated GitHub Issue Responder Bot

[English](README.md) | [中文](README_CN.md)

This repository contains a Python script that automates responses to new issues in specified GitHub repositories using a Language Model (LLM). The bot leverages OpenAI's GPT models, Hugging Face models, or other supported models to generate responses based on the repository's content.

## Features

- **Automated Responses**: Monitors specified GitHub repositories for new issues and automatically generates responses using an LLM.
- **LLM Integration**: Supports OpenAI, Hugging Face, Anthropic, Google, Qianfan, Ollama, or local models for generating responses.
- **Vector Stores**: Builds or loads a vector store of the repository's content for context-aware responses.
- **Environment Configuration**: Utilizes environment variables and a YAML configuration file for flexible setup.
- **Thread-Safe Operations**: Implements locking mechanisms to ensure thread-safe file operations.
- **Extensible Design**: Modular functions allow for easy customization and extension of capabilities.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- Python 3.10 or higher
- Access tokens for GitHub, OpenAI, Hugging Face, or other supported backends

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/GreatV/feifei.git
   cd feifei
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use 'venv\Scripts\activate'
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### Environment Variables

Create a `.env` file in the root directory and add the following variables:

```env
GITHUB_TOKEN=your_github_token
OPENAI_API_KEY=your_openai_api_key  # Optional, if using OpenAI
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token  # Optional, if using Hugging Face
```

### YAML Configuration

Create or modify the `config.yaml` file in the root directory:

```yaml
github_base_url: "https://api.github.com"
llm_backend: "openai"  # Options: 'openai', 'huggingface', 'local', 'anthropic', 'google', 'qianfan', 'ollama'
embedding_backend: "openai"  # Options: 'openai', 'huggingface', 'local'
model_name: "gpt-3.5-turbo"  # LLM model name
embedding_model_name: "sentence-transformers/all-MiniLM-L6-v2"
check_interval: 300  # Time in seconds between checks
repositories:
  yourusername/your-repo:
    start_issue_number: 1
    branch: "main"  # Branch to monitor
    recent_period: 30  # Period in days to consider recent issues
enable_compression: true  # Enable or disable response compression
```

- **github_base_url**: The base URL for GitHub API requests.
- **llm_backend**: The backend to use for the language model.
- **embedding_backend**: The backend to use for embeddings.
- **model_name**: The name of the LLM model.
- **embedding_model_name**: The name of the embeddings model.
- **check_interval**: How often (in seconds) the bot checks for new issues.
- **repositories**: A list of repositories to monitor and their settings.
  - **start_issue_number**: The issue number to start monitoring from.
  - **branch**: The branch to monitor for changes.
  - **recent_period**: The period in days to consider issues and discussions as recent.
- **enable_compression**: Enable or disable response compression.

## Usage

Run the main script:

```bash
python main.py
```

To enable debug mode, use the `--debug` command line argument:

```bash
python main.py --debug
```

The bot will start monitoring the specified repositories for new issues and respond automatically.

## Customization

### Modifying the Prompt Template and Signature

The prompt used to generate responses and the signature appended to each response can now be customized in the `config.yaml` file instead of being hardcoded in the code.

Update the `prompt_template` and `signature` fields in your `config.yaml`:

```yaml
prompt_template: |
  You are a smart assistant capable of answering user questions based on provided documents.

  Question: {input}

  Here are some potentially useful documents:
  {context}

  Please provide a detailed answer based on the above information, in the same language as the question. If possible, reference related Issues or Discussions and provide links.

  Answer:

signature: "\n\n---\n*Response generated by 🤖 feifei-bot | {model_name}*"
```

### Changing the LLM Backend

You can switch between OpenAI, Hugging Face, Anthropic, Google, Qianfan, Ollama, or a local model by updating the `llm_backend` in `config.yaml`:

```yaml
llm_backend: "huggingface"  # or 'local', 'anthropic', 'google', 'qianfan', 'ollama'
```

Ensure that you have the necessary tokens and models installed locally if you choose 'local'.

### Adjusting Retrieval Settings

The number of documents retrieved for context can be adjusted in the `search_kwargs` parameter:

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

2. **Create a New Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Commit Your Changes**

   ```bash
   git commit -am 'Add new feature'
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**

## License

This project is licensed under the [Apache-2.0 license](LICENSE).
