# Contributing to ContextNAFNet Wafer Restoration 🤝

Thank you for your interest in contributing to the **ContextNAFNet Semiconductor Wafer Image Restoration** project!

---

## 🚀 Getting Started

1. **Fork & Clone**:
   ```bash
   git clone https://github.com/kris07hna/AIImageRestoration-Semicon2026.git
   cd AIImageRestoration-Semicon2026
   ```

2. **Create Virtual Environment & Install Dependencies**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Run Test Suite**:
   ```bash
   python -m pytest tests/
   ```

---

## 🛠️ Code Style & Standards

- **Python Version**: Python 3.10+
- **Type Hints**: Use `from __future__ import annotations` and explicit type annotations.
- **Testing**: Ensure all existing unit tests in `tests/` pass cleanly without errors.
- **Formatting**: Adhere to PEP 8 standard style guidelines.

---

## 🧪 Submitting Pull Requests

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Commit your changes: `git commit -m "Add feature description"`
3. Push to your branch: `git push origin feature/your-feature-name`
4. Open a Pull Request on GitHub.
