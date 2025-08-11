# AutoAiTag 🔖

**Intelligent AI-powered metadata generation for Obsidian vaults**

AutoAiTag automatically enhances your Obsidian notes with AI-generated summaries, tags, and intelligent date detection. It's designed to work with local language models (LM Studio) for privacy and cost-effectiveness.

## ✨ Key Features

### 🤖 AI-Powered Metadata Generation
- **Smart Summaries**: Generate concise, contextual summaries (max 50 words) for long notes
- **Intelligent Tagging**: Extract up to 3 relevant keyword tags from note content
- **Date Detection**: Three-tier system for accurate date extraction:
  1. **AI Detection**: LLM extracts dates from note content with confidence scoring
  2. **Filename Parsing**: Extract dates from filenames like `(2024-01-15)`
  3. **File System Fallback**: Use file creation date as last resort

### 🛡️ Vault-Safe Processing
- **Non-destructive**: Preserves all existing frontmatter and metadata
- **Smart Merging**: Combines new AI tags with existing ones
- **Idempotent**: Uses `autoAiTag` flag to prevent re-processing
- **Atomic Writes**: Safe file modifications with backup protection

### ⚡ Performance & Scalability
- **Multithreaded Processing**: Configurable worker threads for large vaults
- **Concurrency Control**: Limit simultaneous LLM requests to prevent server overload
- **Progress Tracking**: Real-time feedback and detailed error logging
- **Batch Processing**: Handle thousands of notes efficiently

### 🔧 Flexible Configuration
- **Local LLM Support**: Works with LM Studio, Ollama, or any OpenAI-compatible API
- **Customizable Thresholds**: Adjust character limits and processing parameters
- **Environment Variables**: Easy deployment across different systems
- **Debug Mode**: Detailed logging for troubleshooting

## 🚀 Installation

### Prerequisites
- **Python 3.7+** (Python 3.8+ recommended)
- **LM Studio** or compatible local LLM server
- **Obsidian vault** with markdown files

### Quick Install
```bash
# Clone the repository
git clone https://github.com/yourusername/AutoAiTag.git
cd AutoAiTag

# Install dependencies
pip install -r requirements-minimal.txt

# Run the script
python AutoAiTag.py
```

### Alternative Installation
```bash
# Install dependencies directly
pip install requests PyYAML

# Download and run
wget https://raw.githubusercontent.com/yourusername/AutoAiTag/main/AutoAiTag.py
python AutoAiTag.py
```

## ⚙️ Configuration

### LM Studio Setup
1. **Install LM Studio** from [lmstudio.ai](https://lmstudio.ai)
2. **Download a model** (recommended: Llama2 7B or larger)
3. **Start the server** with OpenAI-compatible API enabled
4. **Update the script** with your server details:

```python
# In AutoAiTag.py, update these settings:
API_URL = "http://localhost:1234/v1/chat/completions"  # Your LM Studio endpoint
MODEL_NAME = "llama2:7b"  # Your preferred model
```

### Key Configuration Options
```python
# Processing Settings
DEFAULT_CHAR_LIMIT = 1000      # Minimum characters for AI processing
DEFAULT_WORKERS = 4            # Number of concurrent file processors
DEFAULT_MAX_CONCURRENT_LLM = 2 # Max simultaneous LLM requests
REQUEST_TIMEOUT = 60           # LLM response timeout (seconds)
```

## 📖 Usage

### Basic Workflow
1. **Run the script**: `python AutoAiTag.py`
2. **Choose mode**:
   - **Dry Run**: Test without modifying files
   - **Write Mode**: Actually update your notes
3. **Configure settings**:
   - Character threshold for AI processing
   - Worker thread count
   - Force processing options
4. **Process your vault**: Let AutoAiTag enhance your notes!

### Example Output
```yaml
---
title: Meeting Notes
Date: 2024-01-15
summary: Team discussed Q1 goals, budget allocation, and timeline for new product launch
tags: [meeting, planning, budget, product-launch]
wordCount: 1250
autoAiTag: true
---
```

### Processing Modes

#### 🔍 Dry Run Mode
- **Purpose**: Test configuration and see what would happen
- **Output**: `metadata_dryrun.json` with planned changes
- **Safety**: No files are modified
- **Use case**: First-time setup, testing, auditing

#### ✏️ Write Mode
- **Purpose**: Actually enhance your notes with metadata
- **Output**: Modified markdown files with new frontmatter
- **Safety**: Creates backups, preserves existing content
- **Use case**: Production use, bulk processing

## 📊 Error Handling & Logging

### Error Logs
AutoAiTag generates detailed error logs for troubleshooting:
- **Filename**: `(YYYY-MM-DD HH-MM-SS) AutoAiTag Log.json`
- **Contents**: File paths, error messages, timestamps
- **Location**: Same directory as your vault

### Common Issues & Solutions
- **LLM Connection Failed**: Check LM Studio server status and API URL
- **Permission Errors**: Ensure write access to vault directory
- **Memory Issues**: Reduce worker thread count
- **Slow Processing**: Increase LLM concurrency limit (if server can handle it)

## 🔧 Advanced Usage

### Command Line Options
```bash
# Process specific subdirectory
python AutoAiTag.py
# Then enter: ./DailyNotes

# Force reprocessing of all files
python AutoAiTag.py
# Choose Write Mode, then: Force processing = y
```

### Custom Date Formats
AutoAiTag recognizes these filename patterns:
- `Meeting Notes (2024-01-15).md` ✅
- `Journal 2024-01-15.md` ❌ (no parentheses)
- `Notes (2024-1-5).md` ✅ (auto-formats to 2024-01-05)

### Performance Tuning
```python
# For large vaults (1000+ files)
DEFAULT_WORKERS = 8
DEFAULT_MAX_CONCURRENT_LLM = 4

# For slower LLM servers
DEFAULT_MAX_CONCURRENT_LLM = 1
REQUEST_TIMEOUT = 120
```

## 🤝 Contributing

We welcome contributions! Here's how to help:

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest

# Code formatting
black AutoAiTag.py
flake8 AutoAiTag.py
```

### Areas for Improvement
- **Additional LLM providers** (OpenAI, Anthropic, etc.)
- **More metadata fields** (reading time, complexity score)
- **Batch processing** for very large vaults
- **GUI interface** for non-technical users

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LM Studio** for providing local LLM capabilities
- **Obsidian** community for inspiration and feedback
- **Open source contributors** who make tools like this possible

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/tonyknight/AutoAiTag/issues)
- **Wiki**: [Documentation](https://github.com/tonyknight/AutoAiTag/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/tonyknight/AutoAiTag/discussions)

---

**Made with ❤️ for the Obsidian community**

*Transform your vault from a collection of notes into an intelligent, searchable knowledge base with AutoAiTag.*
