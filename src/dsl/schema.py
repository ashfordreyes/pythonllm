"""DSL vocabulary and special tokens.

These are the special tokens to register with the tokenizer via
`tokenizer.add_special_tokens` before fine-tuning the Stage 1 planner model
(see notebooks/02_stage1_finetune.ipynb).
"""

PLAN_START = "<PLAN>"
PLAN_END = "</PLAN>"
STEP_TOKEN = "<STEP>"

SPECIAL_TOKENS = [PLAN_START, PLAN_END, STEP_TOKEN]

# Verbs the DSL supports. Extend this list as the dataset design in the
# literature review calls for more operation types (e.g. TUNE, VISUALIZE).
VERBS = {
    "LOAD",       # LOAD <name> FROM "<path>"
    "SPLIT",      # SPLIT <name> INTO <a>, <b> RATIO <float>
    "TRANSFORM",  # TRANSFORM <name> USING <op> [ON <columns>]
    "TRAIN",      # TRAIN <name> TYPE <model_class> ON <dataset>
    "EVALUATE",   # EVALUATE <name> ON <dataset> METRIC <metric>
    "PREDICT",    # PREDICT <name> ON <dataset>
    "SAVE",       # SAVE <name> TO "<path>"
    # General Python-task verbs -- the training set also covers everyday
    # scripting/debugging/GUI requests, not just the DS/DL model pipeline.
    "FETCH",      # FETCH <name> FROM "<url>"
    "PARSE",      # PARSE <name> USING <format>
    "PRINT",      # PRINT <name>
    "BUILD",      # BUILD <name> TYPE <app_type> USING <libs> [WRAPPING <thing>] [REUSING <parts> FROM "<path>"]
    "DEBUG",      # DEBUG <name> ERROR <error> IN "<path>"
    "READ",       # READ <name> FROM "<path>"
    "EDIT",       # EDIT <name> REMOVE <items> KEEP <items>
    "AUDIT",      # AUDIT <name> FOR <pattern>
    "FIX",        # FIX <name> WHERE <condition> [USING <approach>]
    "FILTER",     # FILTER <name> WHERE <condition> [ON <field>]
    "ADD",        # ADD <thing> TO <target>
    "FINETUNE",   # FINETUNE <name> ON <dataset>
}
