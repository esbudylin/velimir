PYTHON ?= uv run python
LOG_DIR := logs

V ?= 0

ifeq ($(V),1)
Q :=
else
Q := @
endif

export PYTHONPATH = .
export LOG_FILE = $(LOG_DIR)/main.log

define TEST_VERSES
Все неизменно и все изменилось
В утреннем холоде странной свободы.
Долгие годы мне многое снилось,
Вот я проснулся — и где эти годы!

Вот я иду по осеннему полю,
Всё как всегда, и другое, чем прежде:
Точно меня отпустили на волю
И отказали в последней надежде.
endef
export TEST_VERSES

TO_KEBAB = $(subst _,-,$(1))

define SCRIPT_TARGET
$(call TO_KEBAB, $(1)):
	$(Q)LOG_FILE=$(LOG_DIR)/$(1).log $(PYTHON) $(2)/$(1).py
endef

define SCRIPT_TARGET_WITH_TESTS
$(call TO_KEBAB, $(1)):
	$(Q)LOG_FILE=$(LOG_DIR)/$(1).log $(PYTHON) $(2)/$(1).py $(3)

$(call TO_KEBAB, $(1))-test:
	$(Q)LOG_FILE=$(LOG_DIR)/$(1)_test.log $(PYTHON) $(2)/$(1).py --test-run
endef

$(eval $(call SCRIPT_TARGET,markup,entry))

$(eval $(call SCRIPT_TARGET_WITH_TESTS,train,entry,$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE))))

$(eval $(call SCRIPT_TARGET,evaluate_models,entry))

$(eval $(call SCRIPT_TARGET,build_dataset,scripts))

$(eval $(call SCRIPT_TARGET,evaluate_accentuator,scripts))

$(eval $(call SCRIPT_TARGET_WITH_TESTS,build_pos_accent_db,scripts))

$(eval $(call SCRIPT_TARGET_WITH_TESTS,build_grammar_db,scripts))

$(eval $(call SCRIPT_TARGET_WITH_TESTS,build_rhyme_dataset,scripts))

$(eval $(call SCRIPT_TARGET,export_onnx,scripts))

$(eval $(call SCRIPT_TARGET,evaluate_onnx,scripts))

serve-rhymes:
	$(Q)uv run scripts/serve_rhymes.py

$(eval $(call SCRIPT_TARGET_WITH_TESTS,seed_sweep,scripts))

test:
	$(PYTHON) -m unittest discover tests

markup-test:
	echo "$$TEST_VERSES" | LOG_FILE=logs/markup-test.log $(PYTHON) entry/markup.py

