# Changelog

All notable changes to this project will be documented in this file.

## [0.2.1] - 2026-02-15

### 🐛 Bug Fixes

#### Fixed: 中文翻译未保存到 Markdown 文件
- **Problem**: The markdown report showed tool call JSON instead of actual translation
- **Root Cause**: `react_fetch()` was extracting content from `AIMessage` instead of `ToolMessage`
- **Solution**: Modified to correctly extract content from `ToolMessage` in message history
- **Details**: See [BUGFIX.md](BUGFIX.md) for complete analysis

**Modified files**:
- `src/academic_url_agent/main.py` - Updated `react_fetch()` function
- Added `test_extraction.py` - Message extraction test

**Impact**: Now the `report.md` file correctly contains the full Chinese translation.

### 🎉 New Features

#### RFC Document Support
- **Problem**: rfcreader.com URLs returned JavaScript code instead of actual RFC content
- **Solution**: Added `fetch_rfc_direct` tool to fetch directly from IETF official source
- **Details**: See [RFC_SUPPORT.md](RFC_SUPPORT.md) for complete documentation

**New tool**:
- `fetch_rfc_direct(url_or_number)` - Fetch RFC documents from official IETF source
- Auto-extracts RFC number from various URL formats
- Supports: rfcreader.com, rfc-editor.org, tools.ietf.org

**Enhanced detection**:
- `fetch_dynamic` now detects JavaScript code in fetched content
- Agent system prompt updated with RFC-first strategy
- Added `extract_rfc_number()` helper function

**Modified files**:
- `src/academic_url_agent/tools.py` - Added RFC support
- `src/academic_url_agent/graph.py` - Updated agent prompt
- Added `test_rfc.py` - RFC extraction and fetching test

**Impact**: RFC documents (like RFC 7519) now fetch correctly from official source.

**Example**:
```bash
poetry run python -m academic_url_agent.main \
  "http://www.rfcreader.com/#rfc7519"
# Now correctly fetches from: https://www.rfc-editor.org/rfc/rfc7519.txt
```

---

## [0.2.0] - 2026-02-15

### 🎉 New Features

#### Markdown Report Generation
- **Added complete Markdown report generation** (`report.md`)
  - Includes full translation (no truncation)
  - Includes summary, explanations, mindmap
  - Includes original text (collapsible)
  - Includes metadata (timestamp, source URL)
  - Includes table of contents with navigation

#### Enhanced Output
- Console now shows preview (first 2000 chars) with note to check full report
- Clear separation between console output and file output
- Added file output summary at the end

### 🔧 Improvements

- **Lazy Initialization**: LLM instances now use lazy initialization
  - Modules can be imported without API key
  - Better for testing and modular development

- **Better Error Handling**: More informative error messages

### 📝 Documentation

New documentation files:
- `START_HERE.md` - Quick navigation guide
- `QUICKSTART.md` - 3-minute quick start
- `INSTALL.md` - Detailed installation guide
- `USAGE.md` - Comprehensive usage guide
- `PROJECT_SUMMARY.md` - Developer overview
- `FEATURES.md` - Complete feature list
- `REPORT_EXAMPLE.md` - Example report format
- `CHANGELOG.md` - This file

### 🧪 Testing

- Added `test_setup.py` - Verify installation
- Added `test_markdown.py` - Test markdown generation

### 📦 Files Generated

When you run the program, it now generates:
```
report.md         ← Complete Markdown report (NEW!)
mindmap.puml      ← PlantUML script
mindmap.png       ← Mindmap image (optional)
```

### 🔄 API Changes

- `run_pipeline()` now accepts `url` parameter
- `run_pipeline()` now accepts `output_path` parameter (default: `report.md`)
- `run_pipeline()` now returns `markdown_report` in result dict

### 💡 Usage Example

```bash
# Run the program
poetry run python -m academic_url_agent.main "https://example.com/article"

# Output files:
# - report.md       ← Open this for full content
# - mindmap.puml    ← PlantUML source
# - mindmap.png     ← Mindmap image
```

---

## [0.1.0] - 2024-02-13

### Initial Release

#### Features
- ReAct Agent with LangGraph
- Intelligent web scraping (static + dynamic)
- English to Chinese translation with quality check
- Key points summary
- Difficult concepts explanation
- PlantUML mindmap generation
- Local LLM support
- Self-signed certificate handling

---

## Roadmap

### Planned Features

#### v0.3.0
- [ ] PDF extraction tool
- [ ] Batch processing support
- [ ] Custom output templates
- [ ] Export to PDF

#### v0.4.0
- [ ] Human-in-the-loop nodes
- [ ] Content validation nodes
- [ ] Multi-language support (not just EN→CN)
- [ ] Audio/video transcript support

#### v0.5.0
- [ ] Web UI interface
- [ ] API server mode
- [ ] Database persistence
- [ ] Workflow visualization

---

## Breaking Changes

None yet. This is version 0.2.0.

---

## Migration Guide

### From 0.1.0 to 0.2.0

No breaking changes. Existing code continues to work.

**New feature**: The program now automatically generates `report.md` with full content.

If you want to customize the output filename:

```python
# Before
result = run_pipeline(content)

# After (optional)
result = run_pipeline(content, url=url, output_path="my_report.md")
```

---

## Contributors

- Initial implementation based on [Building ReAct agents with LangGraph](https://dylancastillo.co/posts/react-agent-langgraph.html)
- Enhanced with comprehensive documentation and Markdown report generation

---

## Feedback

Found a bug? Have a feature request?
- Open an issue on GitHub
- Check existing documentation first: `START_HERE.md`

---

**Thank you for using Academic URL Agent!** 🎉
