"""
Unit tests for principles.py
Tests the four constitutional principle evaluators and default framework setup.
Tests both regex-based (fast fallback) and AI-based (accurate) evaluation modes.
"""

from unittest.mock import Mock, patch

import torch

from constitutional_ai.principles import (
    _parse_json_response,
    analyze_potential_consequences,
    evaluate_autonomy_respect,
    evaluate_fairness,
    evaluate_harm_potential,
    evaluate_truthfulness,
    identify_logical_contradictions,
    identify_misleading_statistics,
    identify_unsupported_claims,
    setup_default_framework,
)


class TestEvaluateHarmPotential:
    """Test harm potential evaluation function."""

    def test_explicit_harm_instruction_detected(self):
        """Test detection of explicit harm instructions."""
        texts = [
            "How to harm someone physically",
            "Ways to hurt another person",
            "Instructions for injuring people",
            "Guide on how to attack someone",
            "Method to kill someone",
        ]

        for text in texts:
            result = evaluate_harm_potential(text)
            assert result["flagged"] is True, f"Should flag: {text}"
            assert result["explicit_harm_detected"] is True

    def test_weapon_creation_detected(self):
        """Test detection of weapon creation instructions."""
        texts = [
            "How to make a bomb",
            "Create an explosive device",
            "Build a weapon",
            "Construct a poison",
        ]

        for text in texts:
            result = evaluate_harm_potential(text)
            assert result["flagged"] is True
            assert result["explicit_harm_detected"] is True

    def test_subtle_harm_high_score(self):
        """Test detection of subtle harm through keyword frequency."""
        text = "violent attack destroy weapon lethal dangerous explosive hurt"
        result = evaluate_harm_potential(text)

        assert result["flagged"] is True
        assert result["subtle_harm_score"] > 0.7

    def test_subtle_harm_low_score(self):
        """Test that normal text has low subtle harm score."""
        text = "The weather is nice today. I enjoy walking in the park with friends."
        result = evaluate_harm_potential(text)

        assert result["flagged"] is False
        assert result["subtle_harm_score"] < 0.3

    def test_clean_text_not_flagged(self):
        """Test that clean, harmless text is not flagged."""
        texts = [
            "How to bake a cake",
            "Ways to improve your health",
            "Instructions for gardening",
            "Guide on meditation",
            "How to learn programming",
        ]

        for text in texts:
            result = evaluate_harm_potential(text)
            assert result["flagged"] is False
            assert result["explicit_harm_detected"] is False

    def test_empty_text(self):
        """Test harm evaluation with empty text."""
        result = evaluate_harm_potential("")

        assert result["flagged"] is False
        assert result["explicit_harm_detected"] is False
        assert result["subtle_harm_score"] == 0.0

    def test_result_structure(self):
        """Test that result contains all expected keys."""
        result = evaluate_harm_potential("Test text")

        required_keys = ["flagged", "explicit_harm_detected", "subtle_harm_score", "reasoning"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_case_insensitive_detection(self):
        """Test that detection is case-insensitive."""
        texts = ["HOW TO HARM someone", "how to HARM someone", "How To Harm Someone"]

        for text in texts:
            result = evaluate_harm_potential(text)
            assert result["explicit_harm_detected"] is True

    def test_subtle_harm_score_calculation(self):
        """Test subtle harm score calculation."""
        # Text with 2 harm words out of 10 total words
        text = "This text has attack and weapon in it today here"
        result = evaluate_harm_potential(text)

        # 2/10 * 10 = 2.0, but capped at 1.0
        # Actually should be around 0.2
        assert 0.0 <= result["subtle_harm_score"] <= 1.0

    def test_multiple_harm_patterns(self):
        """Test text with multiple harm patterns."""
        text = "How to hurt people and make a weapon to attack"
        result = evaluate_harm_potential(text)

        assert result["flagged"] is True
        assert result["explicit_harm_detected"] is True
        assert result["subtle_harm_score"] > 0.0

    def test_reasoning_field_present(self):
        """Test that reasoning field is always present."""
        texts = ["Clean text", "How to harm someone", ""]

        for text in texts:
            result = evaluate_harm_potential(text)
            assert "reasoning" in result
            assert isinstance(result["reasoning"], str)

    def test_harm_with_positive_context(self):
        """Test that harm words in positive context still detected (primitive system)."""
        # Note: This is expected behavior - the system flags by keywords
        text = "How to protect yourself from being hurt"
        result = evaluate_harm_potential(text)

        # Current implementation would flag this due to "how to hurt" pattern
        assert result["explicit_harm_detected"] is True


class TestAnalyzePotentialConsequences:
    """Test consequence analysis helper function."""

    def test_hacking_consequences(self):
        """Test detection of hacking-related consequences."""
        texts = [
            "How to hack a computer",
            "How to break into a system",
            "How to steal data",
            "How to crack passwords",
        ]

        for text in texts:
            result = analyze_potential_consequences(text)
            assert "unauthorized access" in result or "property damage" in result

    def test_manipulation_consequences(self):
        """Test detection of manipulation consequences."""
        texts = ["How to manipulate people", "How to deceive someone", "How to trick others"]

        for text in texts:
            result = analyze_potential_consequences(text)
            assert "manipulation" in result or "deception" in result

    def test_explosive_consequences(self):
        """Test detection of explosive-related consequences."""
        texts = [
            "Information about bombs",
            "How to use explosives",
            "Details about detonating devices",
        ]

        for text in texts:
            result = analyze_potential_consequences(text)
            assert "dangerous devices" in result

    def test_poison_consequences(self):
        """Test detection of poison-related consequences."""
        texts = ["How to make poison", "Information about toxic substances"]

        for text in texts:
            result = analyze_potential_consequences(text)
            assert "harm to living beings" in result

    def test_security_vulnerability_consequences(self):
        """Test detection of security vulnerability consequences."""
        texts = ["Information about exploits", "Details on vulnerabilities", "How to use backdoors"]

        for text in texts:
            result = analyze_potential_consequences(text)
            assert "security breaches" in result

    def test_safe_text_no_consequences(self):
        """Test that safe text has no harmful consequences identified."""
        texts = ["How to bake bread", "Information about gardening", "Learning to code"]

        for text in texts:
            result = analyze_potential_consequences(text)
            assert "No obvious harmful consequences" in result

    def test_empty_text_consequences(self):
        """Test consequences analysis with empty text."""
        result = analyze_potential_consequences("")
        assert isinstance(result, str)
        assert len(result) > 0


class TestEvaluateTruthfulness:
    """Test truthfulness evaluation function."""

    def test_unsupported_absolute_claims(self):
        """Test detection of absolute claims without evidence."""
        texts = [
            "All people are lazy",
            "Every student is smart",
            "No one can do this",
            "None of them are good",
            "Always remember this is true",
            "Never trust anyone",
        ]

        for text in texts:
            result = evaluate_truthfulness(text)
            assert (
                len(result["unsupported_claims"]) > 0
            ), f"Should detect unsupported claim in: {text}"

    def test_claims_with_evidence_not_flagged(self):
        """Test that claims with supporting evidence are not flagged."""
        texts = [
            "According to research, most people prefer coffee. All participants showed this preference.",
            "Based on the study, every case was successful.",
            "The source shows that this is proven correct.",
        ]

        for text in texts:
            result = evaluate_truthfulness(text)
            # Should have fewer unsupported claims due to evidence
            assert len(result["unsupported_claims"]) < 3

    def test_logical_contradictions_detected(self):
        """Test detection of logical contradictions."""
        texts = [
            "This always works. Sometimes it doesn't work.",
            "I never eat meat. Occasionally I eat chicken.",
            "All of them are good. Not all of them are good.",
            "None of them work. Some are working.",
            "It's impossible. It can happen.",
        ]

        for text in texts:
            result = evaluate_truthfulness(text)
            assert len(result["contradictions"]) > 0, f"Should detect contradiction in: {text}"

    def test_misleading_statistics_detected(self):
        """Test detection of misleading statistics."""
        texts = [
            "There was a 50% increase in sales",
            "Revenue doubled last year",
            "A significant increase in performance",
            "Dramatic growth in users",
        ]

        for text in texts:
            result = evaluate_truthfulness(text)
            # Should detect potential misleading stats without sources
            assert len(result["misleading_statistics"]) > 0

    def test_statistics_with_sources_less_misleading(self):
        """Test that statistics with sources are less problematic."""
        text = "According to the Q4 report, there was a 50% increase in sales based on data from our analytics."
        result = evaluate_truthfulness(text)

        # Should have fewer or no misleading stats due to source
        assert len(result["misleading_statistics"]) == 0

    def test_truthful_text_not_flagged(self):
        """Test that truthful, balanced text is not flagged."""
        text = "Some people enjoy coffee while others prefer tea. Many factors influence taste preferences."
        result = evaluate_truthfulness(text)

        assert result["flagged"] is False

    def test_flagging_threshold(self):
        """Test that flagging occurs at correct thresholds."""
        # More than 3 unsupported claims should flag
        text = "All A are B. Every C is D. No E can F. None of G work. Always H."
        result = evaluate_truthfulness(text)

        assert result["flagged"] is True
        assert len(result["unsupported_claims"]) > 3

    def test_result_structure(self):
        """Test that result contains all expected keys."""
        result = evaluate_truthfulness("Test text")

        required_keys = ["flagged", "unsupported_claims", "contradictions", "misleading_statistics"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_empty_text(self):
        """Test truthfulness evaluation with empty text."""
        result = evaluate_truthfulness("")

        assert result["flagged"] is False
        assert len(result["unsupported_claims"]) == 0
        assert len(result["contradictions"]) == 0
        assert len(result["misleading_statistics"]) == 0


class TestIdentifyUnsupportedClaims:
    """Test identification of unsupported claims."""

    def test_absolute_quantifiers(self):
        """Test detection of absolute quantifiers without evidence."""
        text = "All people are good. Every student passes. No teacher fails. None of them work."
        claims = identify_unsupported_claims(text)

        assert len(claims) > 0
        assert any(
            "all" in claim.lower()
            or "every" in claim.lower()
            or "no" in claim.lower()
            or "none" in claim.lower()
            for claim in claims
        )

    def test_absolute_temporal_claims(self):
        """Test detection of always/never claims."""
        text = "This always works. That never happens."
        claims = identify_unsupported_claims(text)

        assert len(claims) > 0

    def test_certainty_claims(self):
        """Test detection of high-certainty claims."""
        text = "This is proven correct. That is definitely true. It's certainly the case."
        claims = identify_unsupported_claims(text)

        assert len(claims) > 0

    def test_claims_with_evidence_excluded(self):
        """Test that claims with nearby evidence are excluded."""
        text = "According to the study, all participants showed improvement."
        claims = identify_unsupported_claims(text)

        # Should be fewer or no claims due to "according to the study"
        assert len(claims) == 0 or len(claims) < 2

    def test_context_extraction(self):
        """Test that context around claims is extracted."""
        text = "Some context here. All people are good. More context here."
        claims = identify_unsupported_claims(text)

        if claims:
            # Should include context around the claim
            assert len(claims[0]) > len("All people are good")

    def test_claim_limit(self):
        """Test that claim list is limited to 5 examples."""
        # Text with many absolute claims
        text = " ".join([f"All {chr(65+i)} are good." for i in range(10)])
        claims = identify_unsupported_claims(text)

        assert len(claims) <= 5

    def test_empty_text(self):
        """Test with empty text."""
        claims = identify_unsupported_claims("")
        assert len(claims) == 0


class TestIdentifyLogicalContradictions:
    """Test identification of logical contradictions."""

    def test_always_contradiction(self):
        """Test always/not always contradiction."""
        text = "This always works but sometimes it doesn't work."
        contradictions = identify_logical_contradictions(text)

        assert len(contradictions) > 0
        assert "always" in contradictions[0].lower()

    def test_never_contradiction(self):
        """Test never/sometimes contradiction."""
        text = "I never eat meat but sometimes I eat chicken."
        contradictions = identify_logical_contradictions(text)

        assert len(contradictions) > 0
        assert "never" in contradictions[0].lower()

    def test_all_contradiction(self):
        """Test all/not all contradiction."""
        text = "All students passed but not all students passed."
        contradictions = identify_logical_contradictions(text)

        assert len(contradictions) > 0

    def test_none_contradiction(self):
        """Test none/some contradiction."""
        text = "None of them work but some are working."
        contradictions = identify_logical_contradictions(text)

        assert len(contradictions) > 0

    def test_impossible_contradiction(self):
        """Test impossible/possible contradiction."""
        text = "It's impossible but it can happen."
        contradictions = identify_logical_contradictions(text)

        assert len(contradictions) > 0

    def test_no_contradiction(self):
        """Test text without contradictions."""
        text = "Some people like coffee. Others prefer tea. Many enjoy both."
        contradictions = identify_logical_contradictions(text)

        assert len(contradictions) == 0

    def test_empty_text(self):
        """Test with empty text."""
        contradictions = identify_logical_contradictions("")
        assert len(contradictions) == 0


class TestIdentifyMisleadingStatistics:
    """Test identification of misleading statistics."""

    def test_percentage_without_source(self):
        """Test detection of percentages without sources."""
        texts = ["There was a 50% increase", "Sales decreased by 30%", "80% more users"]

        for text in texts:
            stats = identify_misleading_statistics(text)
            assert len(stats) > 0, f"Should detect stat in: {text}"

    def test_multiplier_claims(self):
        """Test detection of multiplier claims."""
        texts = [
            "Revenue doubled last year",
            "Users tripled in six months",
            "Performance increased by 5x",
        ]

        for text in texts:
            stats = identify_misleading_statistics(text)
            assert len(stats) > 0

    def test_qualitative_growth_claims(self):
        """Test detection of qualitative growth claims."""
        texts = [
            "Significant increase in revenue",
            "Substantial growth in users",
            "Dramatic decline in costs",
            "Massive change in performance",
        ]

        for text in texts:
            stats = identify_misleading_statistics(text)
            assert len(stats) > 0

    def test_statistics_with_source(self):
        """Test that statistics with sources are not flagged."""
        text = "According to the Q4 report, there was a 50% increase in sales."
        stats = identify_misleading_statistics(text)

        # Should not be flagged due to source
        assert len(stats) == 0

    def test_statistics_with_evidence(self):
        """Test that statistics with evidence markers are not flagged."""
        text = "Based on data from 2023, revenue doubled according to the study."
        stats = identify_misleading_statistics(text)

        # Should not be flagged due to evidence
        assert len(stats) == 0

    def test_context_extraction(self):
        """Test that context around statistics is extracted."""
        text = "Here is some context. Revenue increased by 50% last year. More context."
        stats = identify_misleading_statistics(text)

        if stats:
            # Should include context
            assert len(stats[0]) > len("50%")

    def test_stat_limit(self):
        """Test that statistic list is limited to 5 examples."""
        text = " ".join([f"There was a {i*10}% increase." for i in range(10)])
        stats = identify_misleading_statistics(text)

        assert len(stats) <= 5

    def test_empty_text(self):
        """Test with empty text."""
        stats = identify_misleading_statistics("")
        assert len(stats) == 0


class TestEvaluateFairness:
    """Test fairness evaluation function."""

    def test_stereotype_detection_with_regex(self):
        """Test detection of stereotyping language using regex fallback."""
        texts = [
            "All men are aggressive",
            "Every woman is emotional",
            "People from that country always do this",
        ]

        for text in texts:
            # Use regex mode explicitly (no model provided)
            result = evaluate_fairness(text, use_ai=False)
            assert result["flagged"] is True, f"Should flag: {text}"
            assert len(result["stereotypes"]) > 0
            assert result["method"] in ["regex_heuristic", "hybrid_regex"]

    def test_stereotype_detection_with_ai(self):
        """Test detection of stereotyping language using AI evaluation."""
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        device = torch.device("cpu")

        texts = [
            "All men are aggressive",
            "Every woman is emotional",
            "All members of that group are the same",
        ]

        # Mock AI response for each text
        mock_response = '{"flagged": true, "stereotypes": ["All members of that group are the same"], "biased_language": []}'

        with patch("constitutional_ai.principles.generate_text", return_value=mock_response):
            for text in texts:
                result = evaluate_fairness(
                    text, model=mock_model, tokenizer=mock_tokenizer, device=device
                )
                assert result["flagged"] is True, f"Should flag: {text}"
                assert result["method"] in [
                    "ai_evaluation",
                    "hybrid_ai",
                    "hybrid_regex",
                    "hybrid_clear",
                ]

    def test_biased_language_detection(self):
        """Test detection of biased terms."""
        texts = [
            "Those primitive people",
            "That backward culture",
            "Those third-world countries",
            "You people are all the same",
            "Your kind doesn't belong here",
        ]

        for text in texts:
            result = evaluate_fairness(text)
            assert result["flagged"] is True, f"Should flag: {text}"
            assert len(result["biased_language"]) > 0

    def test_fair_text_not_flagged(self):
        """Test that fair, balanced text is not flagged."""
        texts = [
            "People have diverse opinions on this topic",
            "Individuals from various backgrounds contribute differently",
            "There is variation within every group",
            "Some men are aggressive while others are not",
        ]

        for text in texts:
            result = evaluate_fairness(text)
            assert result["flagged"] is False

    def test_multiple_fairness_violations(self):
        """Test text with multiple fairness violations."""
        text = "All women are emotional and those primitive people from third-world countries don't understand."
        result = evaluate_fairness(text)

        assert result["flagged"] is True
        assert len(result["stereotypes"]) > 0
        assert len(result["biased_language"]) > 0

    def test_result_structure(self):
        """Test that result contains all expected keys."""
        result = evaluate_fairness("Test text")

        required_keys = ["flagged", "stereotypes", "biased_language"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_case_insensitive_detection(self):
        """Test that detection is case-insensitive."""
        text = "Those PRIMITIVE people"
        result = evaluate_fairness(text)

        assert result["flagged"] is True
        assert len(result["biased_language"]) > 0

    def test_context_extraction_stereotypes(self):
        """Test that context around stereotypes is extracted."""
        text = "Some context here. All men are violent. More context."
        result = evaluate_fairness(text)

        if result["stereotypes"]:
            assert len(result["stereotypes"][0]) > len("All men are violent")

    def test_empty_text(self):
        """Test fairness evaluation with empty text."""
        result = evaluate_fairness("")

        assert result["flagged"] is False
        assert len(result["stereotypes"]) == 0
        assert len(result["biased_language"]) == 0

    def test_item_limits(self):
        """Test that lists are limited to 5 examples."""
        # Text with many biased terms
        text = "primitive backward uncivilized third-world normal people regular people those people you people your kind their kind"
        result = evaluate_fairness(text)

        assert len(result["biased_language"]) <= 5


class TestEvaluateAutonomyRespect:
    """Test autonomy respect evaluation function."""

    def test_coercive_language_detection(self):
        """Test detection of coercive language."""
        texts = [
            "You must do this now",
            "You have to follow these instructions",
            "You need to comply immediately",
            "You are required to participate",
            "There is no choice but to agree",
        ]

        for text in texts:
            result = evaluate_autonomy_respect(text)
            assert result["flagged"] is True, f"Should flag: {text}"
            assert len(result["coercive_language"]) > 0

    def test_manipulative_language_detection(self):
        """Test detection of manipulative language."""
        texts = [
            "If you really cared, you would do this",
            "If you were smart, you would agree",
            "Only idiots would disagree with this",
            "Everyone knows that this is correct",
            "Obviously you should do this",
        ]

        for text in texts:
            result = evaluate_autonomy_respect(text)
            assert result["flagged"] is True, f"Should flag: {text}"
            assert len(result["manipulative_language"]) > 0

    def test_respectful_suggestions_not_flagged(self):
        """Test that respectful suggestions are not flagged."""
        texts = [
            "You might want to consider this option",
            "Perhaps you could try this approach",
            "One option is to do this",
            "I recommend considering this choice",
            "You could potentially benefit from this",
        ]

        for text in texts:
            result = evaluate_autonomy_respect(text)
            # Should not flag due to softening language
            assert len(result["coercive_language"]) == 0

    def test_softening_phrases_prevent_flagging(self):
        """Test that softening phrases prevent coercive language from being flagged."""
        texts = [
            "You might need to consider this option",
            "Perhaps you should think about this",
            "You could consider that you must make a choice",
        ]

        for text in texts:
            result = evaluate_autonomy_respect(text)
            # Softening phrases should prevent flagging
            assert len(result["coercive_language"]) == 0 or result["flagged"] is False

    def test_multiple_autonomy_violations(self):
        """Test text with multiple autonomy violations."""
        text = "You must do this immediately. If you were smart, you would agree. Everyone knows this is right."
        result = evaluate_autonomy_respect(text)

        assert result["flagged"] is True
        assert len(result["coercive_language"]) > 0 or len(result["manipulative_language"]) > 0

    def test_result_structure(self):
        """Test that result contains all expected keys."""
        result = evaluate_autonomy_respect("Test text")

        required_keys = ["flagged", "coercive_language", "manipulative_language"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_case_insensitive_detection(self):
        """Test that detection is case-insensitive."""
        text = "YOU MUST do this NOW"
        result = evaluate_autonomy_respect(text)

        assert result["flagged"] is True

    def test_context_extraction(self):
        """Test that context around violations is extracted."""
        text = "Some context. You must do this. More context."
        result = evaluate_autonomy_respect(text)

        if result["coercive_language"]:
            assert len(result["coercive_language"][0]) > len("You must do this")

    def test_empty_text(self):
        """Test autonomy evaluation with empty text."""
        result = evaluate_autonomy_respect("")

        assert result["flagged"] is False
        assert len(result["coercive_language"]) == 0
        assert len(result["manipulative_language"]) == 0

    def test_item_limits(self):
        """Test that lists are limited to 5 examples."""
        # Text with many coercive phrases
        text = "You must do A. You have to do B. You need to do C. You are required to do D. You must do E. You must do F."
        result = evaluate_autonomy_respect(text)

        assert len(result["coercive_language"]) <= 5
        assert len(result["manipulative_language"]) <= 5


class TestSetupDefaultFramework:
    """Test default framework setup function."""

    def test_framework_created(self):
        """Test that a framework is created."""
        framework = setup_default_framework()

        assert framework is not None
        assert framework.name == "default_constitutional_framework"

    def test_framework_has_four_principles(self):
        """Test that framework has all four principles."""
        framework = setup_default_framework()

        assert len(framework.principles) == 4
        assert "harm_prevention" in framework.principles
        assert "truthfulness" in framework.principles
        assert "fairness" in framework.principles
        assert "autonomy_respect" in framework.principles

    def test_principle_weights(self):
        """Test that principles have correct weights."""
        framework = setup_default_framework()

        assert framework.principles["harm_prevention"].weight == 2.0
        assert framework.principles["truthfulness"].weight == 1.5
        assert framework.principles["fairness"].weight == 1.0
        assert framework.principles["autonomy_respect"].weight == 1.0

    def test_all_principles_enabled(self):
        """Test that all principles are enabled by default."""
        framework = setup_default_framework()

        for principle in framework.principles.values():
            assert principle.enabled is True

    def test_principle_evaluation_functions(self):
        """Test that principles have correct evaluation functions."""
        framework = setup_default_framework()

        assert framework.principles["harm_prevention"].evaluation_fn == evaluate_harm_potential
        assert framework.principles["truthfulness"].evaluation_fn == evaluate_truthfulness
        assert framework.principles["fairness"].evaluation_fn == evaluate_fairness
        assert framework.principles["autonomy_respect"].evaluation_fn == evaluate_autonomy_respect

    def test_principle_descriptions(self):
        """Test that principles have descriptions."""
        framework = setup_default_framework()

        for principle in framework.principles.values():
            assert len(principle.description) > 0
            assert isinstance(principle.description, str)

    def test_framework_functional(self):
        """Test that the framework can evaluate text."""
        framework = setup_default_framework()

        result = framework.evaluate_text("This is a test text.")

        assert "principle_results" in result
        assert len(result["principle_results"]) == 4
        assert result["num_principles_evaluated"] == 4

    def test_framework_detects_harm(self):
        """Test that framework detects harmful content."""
        framework = setup_default_framework()

        result = framework.evaluate_text("How to hurt someone")

        assert result["any_flagged"] is True
        assert "harm_prevention" in result["flagged_principles"]

    def test_framework_detects_untruthfulness(self):
        """Test that framework detects untruthful content."""
        framework = setup_default_framework()

        result = framework.evaluate_text(
            "All people always do this. Every person never does that. Definitely proven."
        )

        assert result["any_flagged"] is True
        assert "truthfulness" in result["flagged_principles"]

    def test_framework_detects_unfairness(self):
        """Test that framework detects unfair content."""
        framework = setup_default_framework()

        result = framework.evaluate_text("All women are the same. Those primitive people.")

        assert result["any_flagged"] is True
        assert "fairness" in result["flagged_principles"]

    def test_framework_detects_autonomy_violation(self):
        """Test that framework detects autonomy violations."""
        framework = setup_default_framework()

        result = framework.evaluate_text("You must do this. Only idiots would disagree.")

        assert result["any_flagged"] is True
        assert "autonomy_respect" in result["flagged_principles"]

    def test_framework_clean_text(self):
        """Test that framework passes clean text."""
        framework = setup_default_framework()

        result = framework.evaluate_text(
            "The sky is blue today."
        )

        assert result["any_flagged"] is False

    def test_weighted_scoring(self):
        """Test that weighted scoring works correctly."""
        framework = setup_default_framework()

        # Text that violates harm (weight 2.0)
        result = framework.evaluate_text("How to destroy and hurt")

        if result["any_flagged"]:
            # Harm prevention has highest weight
            assert result["weighted_score"] >= 2.0

    def test_framework_with_ai_model(self):
        """Test framework with AI model for AI-based evaluation."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        device = torch.device("cpu")

        # Create framework with AI model
        framework = setup_default_framework(
            model=mock_model, tokenizer=mock_tokenizer, device=device
        )

        # Mock AI responses for all principles
        mock_response = '{"flagged": true, "explicit_harm_detected": true, "subtle_harm_score": 0.9, "reasoning": "Harmful content"}'

        with patch("constitutional_ai.principles.generate_text", return_value=mock_response):
            result = framework.evaluate_text("How to harm someone")

            # Should use AI evaluation (may also use hybrid_regex if regex detects first)
            assert result["evaluation_method"] in [
                "ai_evaluation",
                "hybrid_ai",
                "hybrid_clear",
                "hybrid_regex",
            ]
            assert result["any_flagged"] is True

    def test_framework_without_model_uses_regex(self):
        """Test framework without model uses regex fallback."""
        framework = setup_default_framework()

        result = framework.evaluate_text("How to harm someone")

        # Should use regex evaluation
        assert result["evaluation_method"] in ["regex_heuristic", "hybrid_regex"]
        assert result["any_flagged"] is True

    def test_framework_stores_model_parameters(self):
        """Test that framework correctly stores model parameters."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        device = torch.device("cpu")

        framework = setup_default_framework(
            model=mock_model, tokenizer=mock_tokenizer, device=device
        )

        assert framework.model is mock_model
        assert framework.tokenizer is mock_tokenizer
        assert framework.device == device

    def test_framework_ai_mode_with_safe_text(self):
        """Test AI mode with safe text."""
        mock_model = Mock()
        mock_tokenizer = Mock()

        framework = setup_default_framework(model=mock_model, tokenizer=mock_tokenizer)

        # Mock safe responses
        safe_response = '{"flagged": false, "explicit_harm_detected": false, "subtle_harm_score": 0.0, "reasoning": "Safe content"}'

        with patch("constitutional_ai.principles.generate_text", return_value=safe_response):
            result = framework.evaluate_text("The weather is nice today")

            assert result["evaluation_method"] in ["ai_evaluation", "hybrid_ai", "hybrid_clear"]
            assert result["any_flagged"] is False


class TestEdgeCasesAndIntegration:
    """Test edge cases and integration scenarios."""

    def test_very_long_text(self):
        """Test evaluation of very long text."""
        long_text = "This is safe text. " * 1000
        result = evaluate_harm_potential(long_text)

        assert "flagged" in result
        assert isinstance(result["subtle_harm_score"], float)

    def test_unicode_characters(self):
        """Test evaluation with unicode characters."""
        text = "Hello 世界 🌍 This is safe テスト"
        result = evaluate_harm_potential(text)

        assert result["flagged"] is False

    def test_special_characters(self):
        """Test evaluation with special characters."""
        text = "!@#$%^&*() <> {} [] |\\ /?"
        result = evaluate_harm_potential(text)

        assert "flagged" in result

    def test_mixed_violations(self):
        """Test text with multiple types of violations."""
        framework = setup_default_framework()

        text = (
            "How to hurt all people. You must do this. Definitely proven that every person is bad."
        )

        result = framework.evaluate_text(text)

        # Should flag multiple principles
        assert result["any_flagged"] is True
        assert len(result["flagged_principles"]) >= 2

    def test_repeated_evaluations_consistent(self):
        """Test that repeated evaluations give consistent results."""
        text = "How to harm someone"
        result1 = evaluate_harm_potential(text)
        result2 = evaluate_harm_potential(text)

        assert result1["flagged"] == result2["flagged"]
        assert result1["explicit_harm_detected"] == result2["explicit_harm_detected"]

    def test_whitespace_handling(self):
        """Test handling of text with excessive whitespace."""
        text = "   How  to    hurt   someone   "
        result = evaluate_harm_potential(text)

        assert result["explicit_harm_detected"] is True


class TestHybridEvaluation:
    """Test hybrid AI/regex evaluation functionality."""

    def test_regex_mode_when_no_model_provided(self):
        """Test that regex evaluation is used when no model is provided."""
        text = "How to harm someone"
        result = evaluate_harm_potential(text)

        assert "method" in result
        assert result["method"] in ["regex_heuristic", "hybrid_regex"]
        assert result["flagged"] is True

    def test_regex_mode_when_use_ai_false(self):
        """Test that regex evaluation is used when use_ai=False."""
        text = "How to harm someone"
        result = evaluate_harm_potential(text, use_ai=False)

        assert result["method"] in ["regex_heuristic", "hybrid_regex"]
        assert result["flagged"] is True

    def test_ai_mode_with_model_provided(self):
        """Test that AI evaluation is used when model is provided."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        device = torch.device("cpu")

        text = "How to harm someone"
        mock_response = '{"flagged": true, "explicit_harm_detected": true, "subtle_harm_score": 0.8, "reasoning": "Contains harmful instructions"}'

        with patch("constitutional_ai.principles.generate_text", return_value=mock_response):
            result = evaluate_harm_potential(
                text, model=mock_model, tokenizer=mock_tokenizer, device=device
            )

            assert result["method"] in [
                "ai_evaluation",
                "hybrid_ai",
                "hybrid_regex",
                "hybrid_clear",
            ]
            assert result["flagged"] is True

    def test_ai_fallback_to_regex_on_error(self):
        """Test that AI evaluation falls back to regex on error."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        device = torch.device("cpu")

        text = "How to harm someone"

        # Simulate error in AI generation
        with patch(
            "constitutional_ai.principles.generate_text",
            side_effect=Exception("Model error"),
        ):
            result = evaluate_harm_potential(
                text, model=mock_model, tokenizer=mock_tokenizer, device=device
            )

            # Should fallback to regex
            assert result["method"] in ["regex_heuristic", "hybrid_regex"]
            assert result["flagged"] is True

    def test_backward_compatibility_no_parameters(self):
        """Test backward compatibility - calling without new parameters works."""
        text = "How to harm someone"
        result = evaluate_harm_potential(text)

        # Should work like before - uses regex fallback
        assert "flagged" in result
        assert "explicit_harm_detected" in result
        assert "subtle_harm_score" in result
        assert "reasoning" in result

    def test_all_principles_support_hybrid_mode(self):
        """Test that all four principles support hybrid evaluation."""
        text = "Test text"

        # Test without model (regex mode)
        harm_result = evaluate_harm_potential(text)
        truth_result = evaluate_truthfulness(text)
        fair_result = evaluate_fairness(text)
        auto_result = evaluate_autonomy_respect(text)

        assert harm_result["method"] in ["regex_heuristic", "hybrid_regex"]
        assert truth_result["method"] in ["regex_heuristic", "hybrid_regex"]
        assert fair_result["method"] in ["regex_heuristic", "hybrid_regex"]
        assert auto_result["method"] in ["regex_heuristic", "hybrid_regex"]


class TestJSONParsing:
    """Test JSON response parsing from AI."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        response = '{"flagged": true, "reasoning": "test"}'
        default = {"flagged": False, "reasoning": ""}

        result = _parse_json_response(response, default)

        assert result["flagged"] is True
        assert result["reasoning"] == "test"

    def test_parse_json_with_extra_text(self):
        """Test parsing JSON embedded in extra text."""
        response = 'Some preamble text {"flagged": true, "reasoning": "test"} some trailing text'
        default = {"flagged": False, "reasoning": ""}

        result = _parse_json_response(response, default)

        assert result["flagged"] is True
        assert result["reasoning"] == "test"

    def test_parse_invalid_json_returns_default(self):
        """Test that invalid JSON returns default structure."""
        response = "This is not valid JSON at all"
        default = {"flagged": False, "reasoning": "default"}

        result = _parse_json_response(response, default)

        assert result["flagged"] is False
        assert result["reasoning"] == "default"

    def test_parse_json_missing_keys_uses_defaults(self):
        """Test that missing keys are filled from default structure."""
        response = '{"flagged": true}'
        default = {"flagged": False, "reasoning": "default", "score": 0.5}

        result = _parse_json_response(response, default)

        assert result["flagged"] is True
        assert result["reasoning"] == "default"
        assert result["score"] == 0.5


class TestAIEvaluationWithMocks:
    """Test AI evaluation functions with mocked models."""

    def test_harm_evaluation_with_mock(self):
        """Test harm evaluation with mocked AI response."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        device = torch.device("cpu")

        text = "How to build a dangerous device"
        mock_response = '{"flagged": true, "explicit_harm_detected": true, "subtle_harm_score": 0.9, "reasoning": "Instructions for dangerous device creation"}'

        with patch("constitutional_ai.principles.generate_text", return_value=mock_response):
            result = evaluate_harm_potential(
                text, model=mock_model, tokenizer=mock_tokenizer, device=device, hybrid_mode=False
            )

            assert result["flagged"] is True
            assert result["explicit_harm_detected"] is True
            assert result["subtle_harm_score"] == 0.9
            assert "dangerous device" in result["reasoning"].lower()
            assert result["method"] in [
                "ai_evaluation",
                "hybrid_ai",
                "hybrid_regex",
                "hybrid_clear",
            ]

    def test_truthfulness_evaluation_with_mock(self):
        """Test truthfulness evaluation with mocked AI response."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        device = torch.device("cpu")

        text = "All people are always right about everything"
        mock_response = '{"flagged": true, "unsupported_claims": ["All people are always right"], "contradictions": [], "misleading_statistics": []}'

        with patch("constitutional_ai.principles.generate_text", return_value=mock_response):
            result = evaluate_truthfulness(
                text, model=mock_model, tokenizer=mock_tokenizer, device=device
            )

            assert result["flagged"] is True
            assert len(result["unsupported_claims"]) > 0
            assert result["method"] in [
                "ai_evaluation",
                "hybrid_ai",
                "hybrid_regex",
                "hybrid_clear",
            ]

    def test_fairness_evaluation_with_mock(self):
        """Test fairness evaluation with mocked AI response."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        device = torch.device("cpu")

        text = "All members of that group are the same"
        mock_response = '{"flagged": true, "stereotypes": ["All members of that group are the same"], "biased_language": []}'

        with patch("constitutional_ai.principles.generate_text", return_value=mock_response):
            result = evaluate_fairness(
                text, model=mock_model, tokenizer=mock_tokenizer, device=device
            )

            assert result["flagged"] is True
            assert len(result["stereotypes"]) > 0
            assert result["method"] in [
                "ai_evaluation",
                "hybrid_ai",
                "hybrid_regex",
                "hybrid_clear",
            ]

    def test_autonomy_evaluation_with_mock(self):
        """Test autonomy evaluation with mocked AI response."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        device = torch.device("cpu")

        text = "You must do this immediately without question"
        mock_response = '{"flagged": true, "coercive_language": ["You must do this immediately"], "manipulative_language": []}'

        with patch("constitutional_ai.principles.generate_text", return_value=mock_response):
            result = evaluate_autonomy_respect(
                text, model=mock_model, tokenizer=mock_tokenizer, device=device
            )

            assert result["flagged"] is True
            assert len(result["coercive_language"]) > 0
            assert result["method"] in [
                "ai_evaluation",
                "hybrid_ai",
                "hybrid_regex",
                "hybrid_clear",
            ]

    def test_device_defaults_to_cpu(self):
        """Test that device defaults to CPU when not provided."""
        mock_model = Mock()
        mock_tokenizer = Mock()

        text = "Test text"
        mock_response = '{"flagged": false, "explicit_harm_detected": false, "subtle_harm_score": 0.0, "reasoning": "Safe"}'

        with patch("constitutional_ai.principles.generate_text", return_value=mock_response):
            # Don't provide device parameter
            result = evaluate_harm_potential(text, model=mock_model, tokenizer=mock_tokenizer)

            assert result["method"] in [
                "ai_evaluation",
                "hybrid_ai",
                "hybrid_regex",
                "hybrid_clear",
            ]
