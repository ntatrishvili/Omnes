# Expression layer — remaining issues

Issues identified during refactoring analysis that are still worth fixing,
independent of the visitor pattern design choice.

---

## 1. `TimeConditionExpression` owns pandas logic

`__time_to_index()` at `relation.py:777` uses `pd.to_datetime` and
`indexer_between_time` directly on the expression class.  The expression
should be pure data (`entity`, `condition`, `start_time`, `end_time`);
time-index computation belongs in the converter.

**Fix**: move the method into `PulpConverter` (or the base `Converter`)
and pass the result to `convert_time_condition_expression`.

---

## 2. `TimeConditionExpression` fragile name-mangled cache

`__between_times_idx` and `__time_set_id` (lines 689-690) are private
mangled attributes that cache index computation keyed by `time_set.hex_id`.
If the same expression is converted with a different `TimeSet` that shares
an ID, the cache silently returns stale results.  Also thread-unsafe and
makes the object stateful in a surprising way.

**Fix**: remove the cache (the computation is cheap — one `indexer_between_time`
call per time step) as a side-effect of moving the logic to the converter
(issue 1).

---

## 3. `AssignmentExpression._convert` duplicates regex

The `_convert` closure inside `AssignmentExpression.convert()` (lines 807-815)
contains the same self-reference regex as `BinaryExpression._parse_self_reference`
(line 592):

```
r"^\$\.(.+?)(\(t([+-]\d+)?\))?$"
```

This is duplicated parsing logic in two places.

**Fix**: extract the regex to a module-level helper or reuse the existing
parser method.

---

## 4. Inconsistent visitor protocol on base `Converter`

`convert_self_reference` and `convert_entity_reference` are defined only on
`PulpConverter`.  They are not on the base `Converter` class, so a new
converter subclass has no compile-time signal that these methods exist.
`convert_binary_expression` has the same issue.

Contrast with `convert_literal` and `convert_time_condition_expression` which
*are* on the base class (raising `NotImplementedError`).

**Fix**: add stubs (or abstract methods) for all expression-visitor methods
on the base `Converter`, making the protocol explicit.

---

## 5. Structural equality for expression trees

No expression class implements structural `__eq__` / `__hash__` because
`__eq__` is overloaded to build comparison expression nodes (line 204).
This makes it impossible to:
- compare two expression trees for equality in tests
- use expressions in sets or as dict keys
- cache or deduplicate expression trees

**Fix**: add a separate `.equals()` method or a utility function that
deep-compares expression trees structurally.
