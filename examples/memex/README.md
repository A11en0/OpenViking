# Memex - Personal Knowledge Assistant

A CLI-based personal knowledge assistant powered by OpenViking.

## Features

- **Knowledge Management**: Add files, directories, URLs to your knowledge base
- **Intelligent Q&A**: RAG-based question answering with multi-turn conversation
- **Knowledge Browsing**: Navigate with L0/L1/L2 context layers (abstract/overview/full)
- **Semantic Search**: Quick and deep search with intent analysis
- **Feishu Integration**: Import documents from Feishu/Lark (optional)

## Quick Start

```bash
# Install dependencies
uv sync

# Copy and configure
cp ov.conf.example ov.conf
# Edit ov.conf with your API keys

# Run Memex
uv run memex
```

## Configuration

Create `ov.conf` from the example:

```json
{
  "embedding": {
    "dense": {
      "api_base": "https://ark.cn-beijing.volces.com/api/v3",
      "api_key": "your-api-key",
      "backend": "volcengine",
      "dimension": "1024",
      "model": "doubao-embedding-vision-250615"
    }
  },
  "vlm": {
    "api_base": "https://ark.cn-beijing.volces.com/api/v3",
    "api_key": "your-api-key",
    "backend": "volcengine",
    "model": "doubao-seed-1-8-251228"
  }
}
```

## Commands

### Knowledge Management
- `/add <path>` - Add file, directory, or URL
- `/rm <uri>` - Remove resource
- `/import <dir>` - Import entire directory

### Browse
- `/ls [uri]` - List directory contents
- `/tree [uri]` - Show directory tree
- `/read <uri>` - Read full content (L2)
- `/abstract <uri>` - Show summary (L0)
- `/overview <uri>` - Show overview (L1)

### Search
- `/find <query>` - Quick semantic search
- `/search <query>` - Deep search with intent analysis
- `/grep <pattern>` - Content pattern search

### Q&A
- `/ask <question>` - Single-turn question
- `/chat` - Toggle multi-turn chat mode
- `/clear` - Clear chat history
- Or just type your question directly!

### Feishu (Optional)
- `/feishu` - Connect to Feishu
- `/feishu-doc <id>` - Import Feishu document
- `/feishu-search <query>` - Search Feishu documents

Set `FEISHU_APP_ID` and `FEISHU_APP_SECRET` environment variables to enable.

### System
- `/stats` - Show knowledge base statistics
- `/info` - Show configuration
- `/help` - Show help
- `/exit` - Exit Memex

## CLI Options

```bash
uv run memex [OPTIONS]

Options:
  --data-path PATH     Data storage path (default: ./memex_data)
  --user USER          User name (default: default)
  --llm-backend NAME   LLM backend: openai or volcengine (default: openai)
  --llm-model MODEL    LLM model name (default: gpt-4o-mini)
```

## Data Storage

Data is stored in `./memex_data/` by default:
- `viking://resources/` - Your knowledge base
- `viking://user/memories/` - User preferences and memories
- `viking://agent/skills/` - Agent skills and memories
