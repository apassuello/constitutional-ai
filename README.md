# Constitutional AI - Research Implementation

A research implementation of **Constitutional AI** methodology - a technique for training safer language models using principle-based evaluation and AI feedback.

> ⚠️ **Status: Experimental Research Code**
> This is a reference implementation extracted from a research project. Use for learning, experimentation, and research purposes only. Not recommended for production use.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 📋 What This Is

This implementation demonstrates the Constitutional AI methodology from Anthropic's research:
- 4 core constitutional principles (harm prevention, truthfulness, fairness, autonomy)
- Critique-revision supervised learning pipeline
- RLAIF training with reward models and PPO
- Hybrid AI/regex-based evaluation

**Intended for:**
- Learning how Constitutional AI works
- Research experimentation
- Prototyping new approaches
- Educational purposes

**NOT intended for:**
- Production systems
- Mission-critical applications
- Deployment without thorough testing and customization

---

## 🚀 Installation

Clone and install from source:

```bash
git clone <your-repository-url>
cd constitutional-ai
pip install -e .
```

**Requirements:**
- Python 3.8+
- PyTorch 1.13+
- Transformers 4.30+
- See `requirements.txt` for full list

---

## 💡 Quick Start

### Basic Evaluation

```python
from constitutional_ai import setup_default_framework

# Create framework with default principles
framework = setup_default_framework()

# Evaluate text
result = framework.evaluate_text("Your text to evaluate")

print(f"Violations found: {result['any_flagged']}")
print(f"Flagged principles: {result['flagged_principles']}")
print(f"Severity score: {result['weighted_score']}")
```

### Training Example

```python
from constitutional_ai import (
    load_model,
    critique_revision_pipeline,
    setup_default_framework,
)

# Load a small model for experimentation
model, tokenizer = load_model("gpt2")

# Set up framework (defaults to regex-based evaluation)
framework = setup_default_framework()

# Generate critique-revision training data
prompts = ["How can I help others?", "Explain photosynthesis"]
training_data = critique_revision_pipeline(
    prompts=prompts,
    model=model,
    tokenizer=tokenizer,
    framework=framework
)

print(f"Generated {len(training_data['training_data'])} training examples")
```

---

## 📖 What is Constitutional AI?

Constitutional AI is a methodology developed by Anthropic for training language models to be helpful, harmless, and honest using AI-generated feedback.

### Phase 1: Supervised Learning (Critique-Revision)
1. Model generates initial response
2. Model critiques response against constitutional principles
3. Model generates improved response based on critique
4. Fine-tune on improved responses

### Phase 2: Reinforcement Learning from AI Feedback (RLAIF)
1. Generate preference pairs using principles
2. Train reward model on preferences
3. Optimize policy with PPO using reward model

**Research Paper:** [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) (Anthropic, 2022)

---

## 🎯 Core Principles

This implementation includes 4 default constitutional principles:

1. **Harm Prevention** (weight: 2.0)
   - Violence, illegal activities, dangerous instructions
   - Cybercrime and malicious content

2. **Truthfulness** (weight: 1.5)
   - Misleading information, false claims
   - Unsupported assertions

3. **Fairness** (weight: 1.0)
   - Bias and stereotypes
   - Equitable treatment

4. **Autonomy Respect** (weight: 1.0)
   - Coercive or manipulative language
   - User agency

You can define custom principles for your specific use case.

---

## 🔧 Key Components

### Framework
- `ConstitutionalFramework` - Manage and evaluate principles
- `ConstitutionalPrinciple` - Define custom principles
- `setup_default_framework()` - Quick setup with 4 core principles

### Evaluation Functions
- `evaluate_harm_potential()` - Detect harmful content
- `evaluate_truthfulness()` - Identify misleading information
- `evaluate_fairness()` - Check for bias
- `evaluate_autonomy_respect()` - Detect manipulation

### Training Pipeline
- `critique_revision_pipeline()` - Phase 1 supervised learning
- `supervised_finetune()` - Fine-tune on improved responses
- `RewardModel` - Train preference-based reward models
- `PPOTrainer` - Phase 2 policy optimization

### Utilities
- `load_model()` - Load HuggingFace models
- `generate_text()` - Text generation with config
- `ConstitutionalDataset` - PyTorch dataset for training

---

## 📁 Project Structure

```
constitutional-ai/
├── constitutional_ai/       # Main package
│   ├── framework.py        # Core framework
│   ├── principles.py       # Evaluation functions
│   ├── critique_revision.py  # Phase 1 pipeline
│   ├── reward_model.py     # Phase 2 reward model
│   ├── ppo_trainer.py      # Phase 2 PPO trainer
│   ├── config.py           # Configuration
│   └── model_utils.py      # Model utilities
├── tests/                  # Test suite (91% passing)
├── examples/               # Usage examples
├── demos/                  # Interactive demos
└── docs/                   # Documentation (in progress)
```

---

## 🎨 Examples & Demos

### Run Examples
```bash
# Basic quickstart
python examples/quickstart.py

# Custom principles
python examples/custom_principles.py

# Different evaluation modes
python examples/evaluation_modes.py
```

### Interactive Web Demo
```bash
python demos/gradio_demo.py
```

Opens a web interface at `http://localhost:7860` where you can:
- Test principle evaluation on custom text
- See real-time violation detection
- Experiment with different evaluation modes

---

## 🧪 Testing

Current test status: **197/216 tests passing (91%)**

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=constitutional_ai --cov-report=term-missing

# Run specific test module
pytest tests/test_framework.py -v
```

**Known Issues:**
- Some AI evaluation tests require mock configuration updates
- Coverage at 45% (ML code inherently harder to test)
- See `TEST_REPORT.md` for detailed status

---

## ⚙️ Configuration

### Evaluation Modes

**1. Regex-based (default, fast)**
```python
framework = setup_default_framework()  # No model required
```

**2. AI-based (requires model)**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

framework = setup_default_framework(model=model, tokenizer=tokenizer)
```

**3. Hybrid mode (recommended)**
Uses regex as safety net, AI for nuanced evaluation.

---

## 📊 Current Status

### ✅ Working
- All core imports and modules
- Framework principle management
- Regex-based evaluation
- Critique-revision pipeline
- Reward model training
- PPO trainer
- Package builds successfully

### ⚠️ In Progress
- AI-based evaluation (some tests need configuration)
- Comprehensive documentation
- Additional examples
- Performance benchmarks

### ❌ Not Available
- PyPI package (install from source only)
- Pre-trained reward models
- Production deployment guides
- Full API documentation

---

## 🤝 Contributing

This is research code shared for educational purposes. Contributions to improve clarity, fix bugs, or add examples are welcome.

Please note:
- APIs may change as this is experimental code
- No formal support or maintenance guarantees
- Use at your own risk

---

## 📄 License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This implementation is inspired by:

**Anthropic's Constitutional AI Research:**
- Paper: [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- Blog: [Constitutional AI](https://www.anthropic.com/index/constitutional-ai-harmlessness-from-ai-feedback)

**Note:** This is an independent implementation for educational purposes, not affiliated with Anthropic.

---

## 📚 Learn More

- `examples/` - Working code examples
- `demos/` - Interactive demonstrations
- `tests/` - Test suite showing usage patterns
- `FIXES_APPLIED.md` - Recent code improvements
- `TEST_REPORT.md` - Detailed test status

---

## ⚠️ Important Disclaimers

1. **Experimental Code:** This is research/educational code, not production-ready
2. **No Warranties:** Use at your own risk, see LICENSE for details
3. **Not Affiliated:** Independent implementation, not officially from Anthropic
4. **Active Development:** APIs and behavior may change
5. **Testing Required:** Thoroughly test before any serious use

---

**Questions or issues?** Check existing issues or open a new one.

**Want to understand the code?** Start with `examples/quickstart.py` and read the inline documentation.
