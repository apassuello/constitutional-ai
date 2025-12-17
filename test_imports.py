#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import sys
from typing import List, Tuple

def test_import(module_path: str, items: List[str] = None) -> Tuple[bool, str]:
    """Test importing a module or specific items from it"""
    try:
        if items:
            # Test importing specific items
            exec(f"from {module_path} import {', '.join(items)}")
            return True, f"✅ {module_path}: {', '.join(items)}"
        else:
            # Test importing the whole module
            exec(f"import {module_path}")
            return True, f"✅ {module_path}"
    except Exception as e:
        return False, f"❌ {module_path}: {str(e)}"

def main():
    print("=" * 80)
    print("CONSTITUTIONAL AI - IMPORT TESTING")
    print("=" * 80)
    print()

    results = []

    # Test 1: Core framework
    print("1. Testing Core Framework...")
    print("-" * 80)
    success, msg = test_import("constitutional_ai.framework",
                                ["ConstitutionalFramework", "ConstitutionalPrinciple"])
    results.append((success, msg))
    print(msg)
    print()

    # Test 2: Principle evaluators
    print("2. Testing Principle Evaluators...")
    print("-" * 80)
    success, msg = test_import("constitutional_ai.principles", [
        "evaluate_harm_potential",
        "evaluate_truthfulness",
        "evaluate_fairness",
        "evaluate_autonomy_respect",
        "setup_default_framework"
    ])
    results.append((success, msg))
    print(msg)
    print()

    # Test 3: Critique-Revision Pipeline
    print("3. Testing Critique-Revision Pipeline...")
    print("-" * 80)
    success, msg = test_import("constitutional_ai.critique_revision", [
        "generate_critique",
        "generate_revision",
        "critique_revision_pipeline",
        "supervised_finetune",
        "ConstitutionalDataset"
    ])
    results.append((success, msg))
    print(msg)
    print()

    # Test 4: Preference Comparison
    print("4. Testing Preference Comparison...")
    print("-" * 80)
    success, msg = test_import("constitutional_ai.preference_comparison", [
        "generate_comparison",
        "extract_preference",
        "generate_preference_pairs"
    ])
    results.append((success, msg))
    print(msg)
    print()

    # Test 5: Reward Model
    print("5. Testing Reward Model...")
    print("-" * 80)
    success, msg = test_import("constitutional_ai.reward_model", [
        "RewardModel",
        "train_reward_model",
        "compute_reward_loss",
        "RewardModelTrainer"
    ])
    results.append((success, msg))
    print(msg)
    print()

    # Test 6: PPO Trainer
    print("6. Testing PPO Trainer...")
    print("-" * 80)
    success, msg = test_import("constitutional_ai.ppo_trainer", ["PPOTrainer"])
    results.append((success, msg))
    print(msg)
    print()

    # Test 7: RLAIF Trainer
    print("7. Testing RLAIF Trainer...")
    print("-" * 80)
    success, msg = test_import("constitutional_ai.trainer", ["RLAIFTrainer"])
    results.append((success, msg))
    print(msg)
    print()

    # Test 8: Pipeline
    print("8. Testing Pipeline...")
    print("-" * 80)
    success, msg = test_import("constitutional_ai.pipeline", ["ConstitutionalPipeline"])
    results.append((success, msg))
    print(msg)
    print()

    # Test 9: Safety Components
    print("9. Testing Safety Components...")
    print("-" * 80)
    success, msg = test_import("constitutional_ai.evaluator", ["ConstitutionalSafetyEvaluator"])
    results.append((success, msg))
    print(msg)

    success, msg = test_import("constitutional_ai.filter", ["ConstitutionalSafetyFilter"])
    results.append((success, msg))
    print(msg)
    print()

    # Test 10: Configuration
    print("10. Testing Configuration...")
    print("-" * 80)
    success, msg = test_import("constitutional_ai.config", [
        "ConstitutionalTrainingConfig",
        "get_default_config",
        "get_strict_config",
        "get_rlaif_config",
        "get_lightweight_config",
        "get_harm_focused_config"
    ])
    results.append((success, msg))
    print(msg)
    print()

    # Test 11: Model Utilities
    print("11. Testing Model Utilities...")
    print("-" * 80)
    success, msg = test_import("constitutional_ai.model_utils", [
        "load_model",
        "generate_text",
        "GenerationConfig"
    ])
    results.append((success, msg))
    print(msg)
    print()

    # Test 12: Data Utilities
    print("12. Testing Data Utilities...")
    print("-" * 80)
    success, msg = test_import("constitutional_ai.data_utils", ["create_default_prompts"])
    results.append((success, msg))
    print(msg)
    print()

    # Test 13: Main package import
    print("13. Testing Main Package Import...")
    print("-" * 80)
    success, msg = test_import("constitutional_ai")
    results.append((success, msg))
    print(msg)
    print()

    # Test 14: HuggingFace API Evaluator (optional)
    print("14. Testing HuggingFace API Evaluator (optional)...")
    print("-" * 80)
    success, msg = test_import("constitutional_ai.hf_api_evaluator", ["HuggingFaceAPIEvaluator"])
    results.append((success, msg))
    print(msg)
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total = len(results)
    passed = sum(1 for success, _ in results if success)
    failed = total - passed

    print(f"Total tests: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print()

    if failed > 0:
        print("Failed imports:")
        for success, msg in results:
            if not success:
                print(f"  {msg}")
        sys.exit(1)
    else:
        print("All imports successful! 🎉")
        sys.exit(0)

if __name__ == "__main__":
    main()
