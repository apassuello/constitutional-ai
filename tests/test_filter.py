"""
Unit tests for filter.py
Tests the ConstitutionalSafetyFilter class including initialization,
input validation, output filtering, and all regex transformation methods.
"""

from unittest.mock import Mock

from constitutional_ai.filter import ConstitutionalSafetyFilter
from constitutional_ai.framework import ConstitutionalFramework
from constitutional_ai.principles import setup_default_framework


class TestConstitutionalSafetyFilterInit:
    """Test ConstitutionalSafetyFilter initialization."""

    def test_init_with_default_framework(self):
        """Test initialization with default framework (None parameter)."""
        filter_instance = ConstitutionalSafetyFilter()

        assert filter_instance.constitutional_framework is not None
        assert isinstance(filter_instance.constitutional_framework, ConstitutionalFramework)
        assert filter_instance.constitutional_framework.name == "default_constitutional_framework"
        assert filter_instance.base_safety_evaluator is None
        assert filter_instance.strict_mode is False

    def test_init_with_custom_framework(self):
        """Test initialization with custom framework."""
        custom_framework = setup_default_framework()
        custom_framework.name = "custom_test_framework"

        filter_instance = ConstitutionalSafetyFilter(constitutional_framework=custom_framework)

        assert filter_instance.constitutional_framework is custom_framework
        assert filter_instance.constitutional_framework.name == "custom_test_framework"

    def test_init_with_base_safety_evaluator(self):
        """Test initialization with base safety evaluator."""
        mock_evaluator = Mock()
        filter_instance = ConstitutionalSafetyFilter(base_safety_evaluator=mock_evaluator)

        assert filter_instance.base_safety_evaluator is mock_evaluator

    def test_init_without_base_safety_evaluator(self):
        """Test initialization without base safety evaluator."""
        filter_instance = ConstitutionalSafetyFilter()

        assert filter_instance.base_safety_evaluator is None

    def test_init_strict_mode_true(self):
        """Test initialization with strict_mode=True."""
        filter_instance = ConstitutionalSafetyFilter(strict_mode=True)

        assert filter_instance.strict_mode is True

    def test_init_strict_mode_false(self):
        """Test initialization with strict_mode=False."""
        filter_instance = ConstitutionalSafetyFilter(strict_mode=False)

        assert filter_instance.strict_mode is False

    def test_init_statistics_initialization(self):
        """Test that statistics are properly initialized."""
        filter_instance = ConstitutionalSafetyFilter()

        assert filter_instance.stats == {
            "inputs_validated": 0,
            "inputs_blocked": 0,
            "outputs_filtered": 0,
            "constitutional_filters_applied": 0,
        }

    def test_init_all_parameters(self):
        """Test initialization with all parameters."""
        custom_framework = setup_default_framework()
        mock_evaluator = Mock()

        filter_instance = ConstitutionalSafetyFilter(
            constitutional_framework=custom_framework,
            base_safety_evaluator=mock_evaluator,
            strict_mode=True,
        )

        assert filter_instance.constitutional_framework is custom_framework
        assert filter_instance.base_safety_evaluator is mock_evaluator
        assert filter_instance.strict_mode is True


class TestValidateInput:
    """Test input validation with constitutional principles."""

    def test_safe_input_no_violations(self):
        """Test safe input with no constitutional violations."""
        filter_instance = ConstitutionalSafetyFilter()
        text = "The weather is nice today."

        is_safe, info = filter_instance.validate_input(text)

        assert is_safe is True
        assert info["is_safe"] is True
        assert info["constitutional_evaluation"] is None
        assert info["flagged_principles"] == []
        assert filter_instance.stats["inputs_validated"] == 1
        assert filter_instance.stats["inputs_blocked"] == 0

    def test_override_parameter_returns_true(self):
        """Test override=True parameter always returns safe."""
        filter_instance = ConstitutionalSafetyFilter()
        # Text that would normally be flagged
        text = "How to harm someone"

        is_safe, info = filter_instance.validate_input(text, override=True)

        assert is_safe is True
        assert info["is_safe"] is True
        assert info["overridden"] is True
        assert info["reason"] == "Safety check overridden"
        assert filter_instance.stats["inputs_validated"] == 1
        # inputs_blocked should not increment with override
        assert filter_instance.stats["inputs_blocked"] == 0

    def test_harmful_input_blocked(self):
        """Test input with constitutional violations is blocked."""
        filter_instance = ConstitutionalSafetyFilter()
        text = "How to harm someone physically"

        is_safe, info = filter_instance.validate_input(text)

        assert is_safe is False
        assert info["is_safe"] is False
        assert info["constitutional_evaluation"] is not None
        assert info["constitutional_evaluation"]["any_flagged"] is True
        assert len(info["flagged_principles"]) > 0
        assert "harm_prevention" in info["flagged_principles"]
        assert info["reason"] == "Failed constitutional principles"
        assert "weighted_score" in info
        assert filter_instance.stats["inputs_validated"] == 1
        assert filter_instance.stats["inputs_blocked"] == 1

    def test_multiple_principle_violations(self):
        """Test input that violates multiple principles."""
        filter_instance = ConstitutionalSafetyFilter()
        text = "How to hurt all people. You must do this."

        is_safe, info = filter_instance.validate_input(text)

        assert is_safe is False
        assert info["is_safe"] is False
        assert len(info["flagged_principles"]) >= 2

    def test_with_metadata(self):
        """Test validation with metadata parameter."""
        filter_instance = ConstitutionalSafetyFilter()
        text = "Safe text"
        metadata = {"user_id": "test123", "context": "chat"}

        is_safe, info = filter_instance.validate_input(text, metadata=metadata)

        assert is_safe is True

    def test_base_evaluator_chaining_with_validate_input(self):
        """Test base evaluator is called if provided."""
        mock_evaluator = Mock()
        mock_evaluator.validate_input.return_value = (
            True,
            {"is_safe": True, "base_check": "passed"},
        )

        filter_instance = ConstitutionalSafetyFilter(base_safety_evaluator=mock_evaluator)
        text = "Test text"

        is_safe, info = filter_instance.validate_input(text)

        mock_evaluator.validate_input.assert_called_once_with(text, None, False)
        assert is_safe is True

    def test_base_evaluator_without_validate_input_method(self):
        """Test graceful handling when base evaluator lacks validate_input."""
        mock_evaluator = Mock(spec=[])  # No methods

        filter_instance = ConstitutionalSafetyFilter(base_safety_evaluator=mock_evaluator)
        text = "Test text"

        # Should not raise error
        is_safe, info = filter_instance.validate_input(text)

        assert is_safe is True

    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        filter_instance = ConstitutionalSafetyFilter()

        # Validate safe input
        filter_instance.validate_input("Safe text")
        assert filter_instance.stats["inputs_validated"] == 1
        assert filter_instance.stats["inputs_blocked"] == 0

        # Validate harmful input
        filter_instance.validate_input("How to harm someone")
        assert filter_instance.stats["inputs_validated"] == 2
        assert filter_instance.stats["inputs_blocked"] == 1

        # Validate another safe input
        filter_instance.validate_input("Another safe text")
        assert filter_instance.stats["inputs_validated"] == 3
        assert filter_instance.stats["inputs_blocked"] == 1


class TestFilterOutput:
    """Test output filtering with constitutional principles."""

    def test_output_with_no_issues(self):
        """Test output filtering with clean text."""
        filter_instance = ConstitutionalSafetyFilter()
        text = "The weather is nice today."

        filtered_text, info = filter_instance.filter_output(text)

        assert filtered_text == text
        assert info["was_filtered"] is False
        assert info["constitutional_evaluation"] is not None
        assert info["constitutional_evaluation"]["any_flagged"] is False
        assert info["transformations_applied"] == []
        assert filter_instance.stats["outputs_filtered"] == 1
        assert filter_instance.stats["constitutional_filters_applied"] == 0

    def test_output_with_violations_applies_transformations(self):
        """Test output with violations applies transformations."""
        filter_instance = ConstitutionalSafetyFilter()
        text = "How to harm someone physically"

        filtered_text, info = filter_instance.filter_output(text)

        assert filtered_text != text
        assert info["was_filtered"] is True
        assert info["constitutional_evaluation"]["any_flagged"] is True
        assert len(info["transformations_applied"]) > 0
        assert "harm_filtering" in info["transformations_applied"]
        assert "original_length" in info
        assert "final_length" in info
        assert filter_instance.stats["outputs_filtered"] == 1
        assert filter_instance.stats["constitutional_filters_applied"] == 1

    def test_apply_transformations_false(self):
        """Test apply_transformations=False only flags issues."""
        filter_instance = ConstitutionalSafetyFilter()
        text = "How to harm someone physically"

        filtered_text, info = filter_instance.filter_output(text, apply_transformations=False)

        # Text should be unchanged
        assert filtered_text == text
        assert info["was_filtered"] is False
        assert info["constitutional_evaluation"]["any_flagged"] is True
        assert filter_instance.stats["constitutional_filters_applied"] == 0

    def test_base_evaluator_chaining_filter_output(self):
        """Test base evaluator filtering is chained."""
        mock_evaluator = Mock()
        mock_evaluator.filter_output.return_value = (
            "Modified by base",
            {"base_filtered": True},
        )

        filter_instance = ConstitutionalSafetyFilter(base_safety_evaluator=mock_evaluator)
        text = "Test text"

        filtered_text, info = filter_instance.filter_output(text)

        mock_evaluator.filter_output.assert_called_once_with(text, None)
        assert "base_filtered" in info

    def test_base_evaluator_without_filter_output_method(self):
        """Test graceful handling when base evaluator lacks filter_output."""
        mock_evaluator = Mock(spec=[])  # No methods

        filter_instance = ConstitutionalSafetyFilter(base_safety_evaluator=mock_evaluator)
        text = "Test text"

        # Should not raise error
        filtered_text, info = filter_instance.filter_output(text)

        assert filtered_text == text

    def test_output_statistics_tracking(self):
        """Test statistics tracking for output filtering."""
        filter_instance = ConstitutionalSafetyFilter()

        # Filter clean output
        filter_instance.filter_output("Clean text")
        assert filter_instance.stats["outputs_filtered"] == 1
        assert filter_instance.stats["constitutional_filters_applied"] == 0

        # Filter harmful output
        filter_instance.filter_output("How to harm someone")
        assert filter_instance.stats["outputs_filtered"] == 2
        assert filter_instance.stats["constitutional_filters_applied"] == 1

    def test_with_metadata_parameter(self):
        """Test filter_output with metadata."""
        filter_instance = ConstitutionalSafetyFilter()
        text = "Safe text"
        metadata = {"context": "response"}

        filtered_text, info = filter_instance.filter_output(text, metadata=metadata)

        assert filtered_text == text


class TestHarmFiltering:
    """Test _filter_harmful_content method."""

    def test_how_to_harm_pattern(self):
        """Test 'how to harm' pattern replacement."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"explicit_harm_detected": True}

        text = "Here is how to harm someone with a weapon."
        filtered = filter_instance._filter_harmful_content(text, evaluation)

        assert "I cannot provide instructions that could cause harm" in filtered
        assert "how to harm someone" not in filtered.lower()

    def test_ways_to_harm_pattern(self):
        """Test 'ways to harm' pattern replacement."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"explicit_harm_detected": True}

        text = "There are ways to hurt people effectively."
        filtered = filter_instance._filter_harmful_content(text, evaluation)

        assert "I cannot provide methods that could cause harm" in filtered

    def test_make_weapon_pattern(self):
        """Test weapon creation pattern replacement."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"explicit_harm_detected": True}

        text = "How to make a bomb at home."
        filtered = filter_instance._filter_harmful_content(text, evaluation)

        assert "I cannot provide instructions for creating dangerous items" in filtered

    def test_create_poison_pattern(self):
        """Test poison creation pattern replacement."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"explicit_harm_detected": True}

        text = "How to create poison from household items."
        filtered = filter_instance._filter_harmful_content(text, evaluation)

        assert "I cannot provide instructions for creating dangerous items" in filtered

    def test_safe_text_unchanged(self):
        """Test safe text remains unchanged."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"explicit_harm_detected": False}

        text = "How to make a delicious cake."
        filtered = filter_instance._filter_harmful_content(text, evaluation)

        assert filtered == text

    def test_multiple_harmful_patterns(self):
        """Test text with multiple harmful patterns."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"explicit_harm_detected": True}

        text = "How to harm people and make a weapon to attack them."
        filtered = filter_instance._filter_harmful_content(text, evaluation)

        assert "I cannot provide" in filtered
        assert len(filtered) < len(text) or "cannot" in filtered

    def test_case_insensitive_matching(self):
        """Test case-insensitive pattern matching."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"explicit_harm_detected": True}

        text = "HOW TO HARM someone"
        filtered = filter_instance._filter_harmful_content(text, evaluation)

        assert "I cannot provide instructions that could cause harm" in filtered


class TestTruthfulnessFiltering:
    """Test _add_truthfulness_qualifiers method."""

    def test_many_unsupported_claims_adds_disclaimer(self):
        """Test disclaimer added for >2 unsupported claims."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"unsupported_claims": ["claim1", "claim2", "claim3"]}

        text = "Original text here."
        filtered = filter_instance._add_truthfulness_qualifiers(text, evaluation)

        assert (
            "Note: Some statements above may represent opinions or claims that require additional verification"
            in filtered
        )
        assert "Original text here." in filtered

    def test_few_unsupported_claims_no_disclaimer(self):
        """Test no disclaimer for <=2 unsupported claims."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"unsupported_claims": ["claim1", "claim2"]}

        text = "Original text here."
        filtered = filter_instance._add_truthfulness_qualifiers(text, evaluation)

        assert filtered == text

    def test_contradictions_adds_note(self):
        """Test note added for contradictions."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {
            "unsupported_claims": [],
            "contradictions": ["contradiction1"],
        }

        text = "Original text here."
        filtered = filter_instance._add_truthfulness_qualifiers(text, evaluation)

        assert (
            "Please note: There may be some inconsistencies in the information provided above"
            in filtered
        )

    def test_misleading_statistics_adds_note(self):
        """Test note added for >1 misleading statistics."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {
            "unsupported_claims": [],
            "contradictions": [],
            "misleading_statistics": ["stat1", "stat2"],
        }

        text = "Original text here."
        filtered = filter_instance._add_truthfulness_qualifiers(text, evaluation)

        assert "Note: Statistical claims should be verified with original sources" in filtered

    def test_safe_text_unchanged(self):
        """Test safe text remains unchanged."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {
            "unsupported_claims": [],
            "contradictions": [],
            "misleading_statistics": [],
        }

        text = "Original text here."
        filtered = filter_instance._add_truthfulness_qualifiers(text, evaluation)

        assert filtered == text

    def test_no_duplicate_disclaimers(self):
        """Test that disclaimers aren't duplicated."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"unsupported_claims": ["c1", "c2", "c3"]}

        text = "Original text.\n\nNote: Some statements above may represent opinions or claims that require additional verification."
        filtered = filter_instance._add_truthfulness_qualifiers(text, evaluation)

        # Should not add duplicate
        assert filtered == text

    def test_priority_of_notes(self):
        """Test that only one note is added when multiple conditions met."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {
            "unsupported_claims": ["c1", "c2", "c3"],
            "contradictions": ["con1"],
            "misleading_statistics": ["s1", "s2"],
        }

        text = "Original text."
        filtered = filter_instance._add_truthfulness_qualifiers(text, evaluation)

        # Should add unsupported claims disclaimer (first check)
        assert (
            "Note: Some statements above may represent opinions or claims that require additional verification"
            in filtered
        )
        # Should not add other notes (qualifiers_added flag prevents it)
        assert "inconsistencies" not in filtered


class TestBiasFiltering:
    """Test _filter_biased_content method."""

    def test_all_women_softened(self):
        """Test 'all women' generalization is softened."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"stereotypes": ["stereotype1"]}

        text = "All women are emotional."
        filtered = filter_instance._filter_biased_content(text, evaluation)

        assert "some women" in filtered.lower()
        assert "all women" not in filtered.lower()

    def test_every_men_softened(self):
        """Test 'every men' generalization is softened."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"stereotypes": ["stereotype1"]}

        text = "Every man is strong."
        filtered = filter_instance._filter_biased_content(text, evaluation)

        assert "some man" in filtered.lower()

    def test_always_softened_strict_mode(self):
        """Test 'always' softened to 'often' in strict mode."""
        filter_instance = ConstitutionalSafetyFilter(strict_mode=True)
        evaluation = {"stereotypes": ["stereotype1"]}

        text = "They always do this."
        filtered = filter_instance._filter_biased_content(text, evaluation)

        assert "often" in filtered.lower()
        assert "always" not in filtered.lower()

    def test_always_softened_normal_mode(self):
        """Test 'always' softened to 'sometimes' in normal mode."""
        filter_instance = ConstitutionalSafetyFilter(strict_mode=False)
        evaluation = {"stereotypes": ["stereotype1"]}

        text = "They always do this."
        filtered = filter_instance._filter_biased_content(text, evaluation)

        assert "sometimes" in filtered.lower()
        assert "always" not in filtered.lower()

    def test_primitive_replaced(self):
        """Test 'primitive' replaced with 'traditional'."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"biased_language": ["primitive"]}

        text = "Those primitive people."
        filtered = filter_instance._filter_biased_content(text, evaluation)

        assert "traditional" in filtered.lower()
        assert "primitive" not in filtered.lower()

    def test_normal_people_replaced(self):
        """Test 'normal people' replaced with 'most people'."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"biased_language": ["normal people"]}

        text = "Normal people don't do that."
        filtered = filter_instance._filter_biased_content(text, evaluation)

        assert "most people" in filtered.lower()
        assert "normal people" not in filtered.lower()

    def test_uncivilized_strict_mode(self):
        """Test 'uncivilized' replaced with 'non-Western' in strict mode."""
        filter_instance = ConstitutionalSafetyFilter(strict_mode=True)
        evaluation = {"biased_language": ["uncivilized"]}

        text = "Those uncivilized cultures."
        filtered = filter_instance._filter_biased_content(text, evaluation)

        assert "non-Western" in filtered or "non-western" in filtered.lower()

    def test_uncivilized_normal_mode(self):
        """Test 'uncivilized' replaced with 'different' in normal mode."""
        filter_instance = ConstitutionalSafetyFilter(strict_mode=False)
        evaluation = {"biased_language": ["uncivilized"]}

        text = "Those uncivilized cultures."
        filtered = filter_instance._filter_biased_content(text, evaluation)

        assert "different" in filtered.lower()

    def test_safe_text_unchanged(self):
        """Test safe text remains unchanged."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"stereotypes": [], "biased_language": []}

        text = "Some people prefer coffee."
        filtered = filter_instance._filter_biased_content(text, evaluation)

        assert filtered == text

    def test_case_insensitive_replacement(self):
        """Test case-insensitive bias replacement."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"biased_language": ["PRIMITIVE"]}

        text = "Those PRIMITIVE cultures."
        filtered = filter_instance._filter_biased_content(text, evaluation)

        assert "traditional" in filtered.lower()


class TestAutonomyFiltering:
    """Test _filter_autonomy_violations method."""

    def test_must_softened(self):
        """Test 'must' softened to 'may want to'."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"coercive_language": ["must"]}

        text = "You must do this now."
        filtered = filter_instance._filter_autonomy_violations(text, evaluation)

        assert "may want to" in filtered.lower()
        assert "must" not in filtered.lower()

    def test_have_to_softened(self):
        """Test 'have to' softened to 'may want to'."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"coercive_language": ["have to"]}

        text = "You have to comply."
        filtered = filter_instance._filter_autonomy_violations(text, evaluation)

        assert "may want to" in filtered.lower()

    def test_required_to_replaced(self):
        """Test 'required to' replaced with 'recommended to'."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"coercive_language": ["required"]}

        text = "You are required to participate."
        filtered = filter_instance._filter_autonomy_violations(text, evaluation)

        assert "recommended to" in filtered.lower()

    def test_if_you_really_cared_replaced(self):
        """Test 'if you really cared' replaced with 'if you are interested'."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"manipulative_language": ["if you really cared"]}

        text = "If you really cared about this, you would agree."
        filtered = filter_instance._filter_autonomy_violations(text, evaluation)

        assert "if you are interested" in filtered.lower()

    def test_if_you_were_smart_replaced(self):
        """Test 'if you were smart' replaced with 'one smart approach'."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"manipulative_language": ["if you were smart"]}

        text = "If you were smart, you would do this."
        filtered = filter_instance._filter_autonomy_violations(text, evaluation)

        assert "one smart approach" in filtered.lower()

    def test_obviously_replaced(self):
        """Test 'obviously' replaced with 'arguably'."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"manipulative_language": ["obviously"]}

        text = "Obviously this is correct."
        filtered = filter_instance._filter_autonomy_violations(text, evaluation)

        assert "arguably" in filtered.lower()
        assert "obviously" not in filtered.lower()

    def test_safe_text_unchanged(self):
        """Test safe text remains unchanged."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"coercive_language": [], "manipulative_language": []}

        text = "You might want to consider this option."
        filtered = filter_instance._filter_autonomy_violations(text, evaluation)

        assert filtered == text

    def test_multiple_coercive_patterns(self):
        """Test multiple coercive patterns replaced."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"coercive_language": ["must", "required"]}

        text = "You must do this and you are required to comply."
        filtered = filter_instance._filter_autonomy_violations(text, evaluation)

        assert "may want to" in filtered.lower()
        assert "recommended to" in filtered.lower()

    def test_case_insensitive_replacement(self):
        """Test case-insensitive autonomy violation replacement."""
        filter_instance = ConstitutionalSafetyFilter()
        evaluation = {"coercive_language": ["MUST"]}

        text = "You MUST do this."
        filtered = filter_instance._filter_autonomy_violations(text, evaluation)

        assert "may want to" in filtered.lower()


class TestStatistics:
    """Test statistics methods."""

    def test_get_statistics_returns_dict(self):
        """Test get_statistics returns dictionary with stats."""
        filter_instance = ConstitutionalSafetyFilter()

        stats = filter_instance.get_statistics()

        assert isinstance(stats, dict)
        assert "inputs_validated" in stats
        assert "inputs_blocked" in stats
        assert "outputs_filtered" in stats
        assert "constitutional_filters_applied" in stats
        assert "framework_stats" in stats

    def test_get_statistics_includes_framework_stats(self):
        """Test get_statistics includes framework statistics."""
        filter_instance = ConstitutionalSafetyFilter()

        # Trigger some evaluations
        filter_instance.validate_input("Test text")

        stats = filter_instance.get_statistics()

        assert "framework_stats" in stats
        assert isinstance(stats["framework_stats"], dict)

    def test_reset_statistics_clears_counters(self):
        """Test reset_statistics clears all counters."""
        filter_instance = ConstitutionalSafetyFilter()

        # Generate some statistics
        filter_instance.validate_input("How to harm someone")
        filter_instance.filter_output("How to harm someone")

        assert filter_instance.stats["inputs_validated"] > 0
        assert filter_instance.stats["outputs_filtered"] > 0

        # Reset
        filter_instance.reset_statistics()

        assert filter_instance.stats["inputs_validated"] == 0
        assert filter_instance.stats["inputs_blocked"] == 0
        assert filter_instance.stats["outputs_filtered"] == 0
        assert filter_instance.stats["constitutional_filters_applied"] == 0

    def test_reset_statistics_clears_framework_history(self):
        """Test reset_statistics clears framework history."""
        mock_framework = Mock()
        filter_instance = ConstitutionalSafetyFilter(constitutional_framework=mock_framework)

        filter_instance.reset_statistics()

        mock_framework.clear_history.assert_called_once()

    def test_statistics_tracking_across_operations(self):
        """Test statistics are correctly tracked across operations."""
        filter_instance = ConstitutionalSafetyFilter()

        # Validate inputs
        filter_instance.validate_input("Safe text")
        filter_instance.validate_input("How to harm someone")

        # Filter outputs
        filter_instance.filter_output("Safe output")
        filter_instance.filter_output("How to harm someone")

        stats = filter_instance.get_statistics()

        assert stats["inputs_validated"] == 2
        assert stats["inputs_blocked"] == 1
        assert stats["outputs_filtered"] == 2
        assert stats["constitutional_filters_applied"] == 1


class TestIntegrationScenarios:
    """Test integrated filtering scenarios."""

    def test_end_to_end_harm_filtering(self):
        """Test complete harm filtering flow."""
        filter_instance = ConstitutionalSafetyFilter()

        # Input validation
        is_safe, _ = filter_instance.validate_input("How to harm someone")
        assert is_safe is False

        # Output filtering
        harmful_output = "Here is how to harm someone with violence."
        filtered, info = filter_instance.filter_output(harmful_output)

        assert filtered != harmful_output
        assert info["was_filtered"] is True
        assert "harm_filtering" in info["transformations_applied"]

    def test_end_to_end_truthfulness_filtering(self):
        """Test complete truthfulness filtering flow."""
        filter_instance = ConstitutionalSafetyFilter()

        text = (
            "All people always do this. Every person definitely agrees. None disagree. Always true."
        )

        # Should be flagged
        is_safe, _ = filter_instance.validate_input(text)
        assert is_safe is False

        # Should add qualifiers
        filtered, info = filter_instance.filter_output(text)
        assert "Note:" in filtered or "note:" in filtered.lower()

    def test_end_to_end_bias_filtering(self):
        """Test complete bias filtering flow."""
        filter_instance = ConstitutionalSafetyFilter()

        text = "All women are the same. Those primitive people."

        is_safe, _ = filter_instance.validate_input(text)
        assert is_safe is False

        filtered, info = filter_instance.filter_output(text)
        assert "some women" in filtered.lower()
        assert "traditional" in filtered.lower()

    def test_end_to_end_autonomy_filtering(self):
        """Test complete autonomy filtering flow."""
        filter_instance = ConstitutionalSafetyFilter()

        text = "You must do this immediately. Obviously this is correct."

        is_safe, _ = filter_instance.validate_input(text)
        assert is_safe is False

        filtered, info = filter_instance.filter_output(text)
        assert "may want to" in filtered.lower()
        assert "arguably" in filtered.lower()

    def test_multiple_transformations_applied(self):
        """Test multiple transformation types in one output."""
        filter_instance = ConstitutionalSafetyFilter()

        # Text that violates multiple principles for different reasons
        text = "How to harm someone. All people always lie. You must comply with this."

        filtered, info = filter_instance.filter_output(text)

        assert info["was_filtered"] is True
        # Should apply multiple transformations (harm, truthfulness, autonomy)
        assert len(info["transformations_applied"]) >= 1

    def test_strict_mode_differences(self):
        """Test differences between strict and normal mode."""
        normal_filter = ConstitutionalSafetyFilter(strict_mode=False)
        strict_filter = ConstitutionalSafetyFilter(strict_mode=True)

        text = "They always do this."
        evaluation = {"stereotypes": ["always"]}

        normal_filtered = normal_filter._filter_biased_content(text, evaluation)
        strict_filtered = strict_filter._filter_biased_content(text, evaluation)

        # Strict mode uses "often", normal mode uses "sometimes"
        assert "sometimes" in normal_filtered.lower() or "some" in normal_filtered.lower()
        assert "often" in strict_filtered.lower() or "some" in strict_filtered.lower()

    def test_clean_text_full_pipeline(self):
        """Test clean text passes through entire pipeline unchanged."""
        filter_instance = ConstitutionalSafetyFilter()

        text = "The weather is nice today. Many people enjoy sunshine."

        # Should pass validation
        is_safe, _ = filter_instance.validate_input(text)
        assert is_safe is True

        # Should pass filtering unchanged
        filtered, info = filter_instance.filter_output(text)
        assert filtered == text
        assert info["was_filtered"] is False

    def test_override_bypasses_all_checks(self):
        """Test override parameter bypasses all validation."""
        filter_instance = ConstitutionalSafetyFilter()

        dangerous_text = "How to harm and kill people. You must do this."

        is_safe, info = filter_instance.validate_input(dangerous_text, override=True)

        assert is_safe is True
        assert info["overridden"] is True
        # Should not increment blocked counter with override
        assert filter_instance.stats["inputs_blocked"] == 0
