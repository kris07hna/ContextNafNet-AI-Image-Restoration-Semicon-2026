# Technical Documentation Index

Welcome to the comprehensive technical documentation for ContextNAFNet, built for the KLA Problem Statement – AI-Based Restoration of Degraded Images at SEMICON Hackathon 2026.

---

## Documentation Overview

| Document File | Purpose & Contents |
|:---|:---|
| [`README.md`](file:///c:/Users/krish/semicon2026/.claude/worktrees/practical-heisenberg-81daae/README.md) | Primary project homepage with live GIF demo, architecture blueprint, benchmark tables, and quick start instructions. |
| [`documentation/SUBMISSION_GUIDE.md`](file:///c:/Users/krish/semicon2026/.claude/worktrees/practical-heisenberg-81daae/documentation/SUBMISSION_GUIDE.md) | Official KLA Problem Statement submission guide, technical checklist verification, and command signatures. |
| [`documentation/IEEE_Paper_Architecture_Description.md`](file:///c:/Users/krish/semicon2026/.claude/worktrees/practical-heisenberg-81daae/documentation/IEEE_Paper_Architecture_Description.md) | Complete IEEE paper specification with mathematical formulas, SimpleGate activation-free equations, and receptive field calculations. |
| [`reports/PRODUCTION_REPORT.md`](file:///c:/Users/krish/semicon2026/.claude/worktrees/practical-heisenberg-81daae/reports/PRODUCTION_REPORT.md) | Quantitative validation report with PSNR, SSIM, SNR gains, and residual loss heatmaps across test wafers. |
| [`LICENSE`](file:///c:/Users/krish/semicon2026/.claude/worktrees/practical-heisenberg-81daae/LICENSE) | MIT Open Source License. |
| [`CONTRIBUTING.md`](file:///c:/Users/krish/semicon2026/.claude/worktrees/practical-heisenberg-81daae/CONTRIBUTING.md) | Contribution guidelines, environment setup, and pull request workflows. |

---

## Quick Command Reference

```bash
# 1. Official Submission Entrypoint
python run.py <input-dir> <output-dir>

# 2. Evaluation Suite
python scripts/evaluate.py --checkpoint models/best.pt --tta 8

# 3. Single-Sample Inspection & SNR Calculator
python scripts/compare.py --file 002060.npy --checkpoint models/best.pt

# 4. PyTorch Unit Tests
python -m pytest tests/
```
