#!/usr/bin/env python3
"""
Example: Creating Custom Constitutional Principles

This example shows how to create and use custom constitutional principles
beyond the default 4 principles (harm, truthfulness, fairness, autonomy).
"""

from constitutional_ai import (
    ConstitutionalFramework,
    ConstitutionalPrinciple,
)


def evaluate_privacy_respect(text: str, **kwargs) -> dict:
    """
    Custom principle: Privacy Respect

    Checks if text respects user privacy and data protection.
    """
    privacy_keywords = [
        "share your password",
        "send me your credit card",
        "give me your ssn",
        "your personal data",
        "track your location",
    ]

    flagged = any(keyword in text.lower() for keyword in privacy_keywords)

    return {
        "flagged": flagged,
        "reason": "Privacy violation detected" if flagged else "No privacy issues",
        "method": "regex_heuristic",
    }


def evaluate_spam_prevention(text: str, **kwargs) -> dict:
    """
    Custom principle: Spam Prevention

    Detects spam-like content (excessive caps, promotional language, etc.).
    """
    spam_indicators = [
        "buy now",
        "limited time offer",
        "click here",
        "100% free",
        "act now",
    ]

    # Check for all caps (spam indicator)
    all_caps_ratio = sum(1 for c in text if c.isupper()) / (len(text) + 1)

    flagged = (
        all_caps_ratio > 0.5 or
        sum(indicator in text.lower() for indicator in spam_indicators) >= 2
    )

    return {
        "flagged": flagged,
        "reason": "Spam-like content detected" if flagged else "No spam detected",
        "all_caps_ratio": all_caps_ratio,
        "method": "regex_heuristic",
    }


def main():
    """Demonstrate custom principles."""
    print("=" * 70)
    print("Custom Constitutional Principles Example")
    print("=" * 70)

    # Create framework
    framework = ConstitutionalFramework(name="custom_framework")

    # Add custom principles
    privacy_principle = ConstitutionalPrinciple(
        name="privacy_respect",
        description="Respects user privacy and data protection",
        evaluation_fn=evaluate_privacy_respect,
        weight=1.5,
    )

    spam_principle = ConstitutionalPrinciple(
        name="spam_prevention",
        description="Prevents spam and promotional content",
        evaluation_fn=evaluate_spam_prevention,
        weight=1.0,
    )

    framework.add_principle(privacy_principle)
    framework.add_principle(spam_principle)

    print(f"\nFramework: {framework}")
    print(f"Active principles: {framework.get_active_principles()}")

    # Test cases
    test_cases = [
        "Hello! How can I help you today?",
        "BUY NOW!!! CLICK HERE FOR 100% FREE OFFER!!!",
        "Please share your password with me for verification.",
        "Here are some helpful tips for staying safe online.",
    ]

    print("\n" + "=" * 70)
    print("Testing Custom Principles")
    print("=" * 70)

    for i, text in enumerate(test_cases, 1):
        print(f"\n[Test {i}]")
        print(f"Text: {text}")

        result = framework.evaluate_text(text)

        print(f"Flagged: {result['any_flagged']}")
        if result['any_flagged']:
            print(f"Violations: {result['flagged_principles']}")
            for principle_name in result['flagged_principles']:
                principle_result = result['principle_results'][principle_name]
                print(f"  - {principle_name}: {principle_result['reason']}")

    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Create evaluation functions that return dict with 'flagged' key")
    print("  • Wrap functions in ConstitutionalPrinciple with name and weight")
    print("  • Add principles to framework with add_principle()")
    print("  • Use evaluate_text() to check against all principles")


if __name__ == "__main__":
    main()
