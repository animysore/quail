"""Logical node and optimizer extension tests."""

from dataclasses import dataclass, replace
from typing import ClassVar

from quail.logical import ColumnRef, LogicalPlan, Project, Scan
from quail.planner.logical_optimizer import (
    LogicalPlanningContext,
    apply_logical_rules,
)

CONTEXT = LogicalPlanningContext(catalog=None, engine_config=None)


@dataclass(frozen=True)
class TaggedInput:
    """Logical test node defined outside the built in node list."""

    input: object
    tag: str

    type_name: ClassVar[str] = "test.tagged_input"

    def children(self):
        return (self.input,)

    def expressions(self):
        return (self.tag,)

    def output_schema(self):
        return self.input.output_schema()

    def validate(self):
        pass

    def with_children(self, children):
        (child,) = children
        return replace(self, input=child)

    def with_expressions(self, expressions):
        (tag,) = expressions
        return replace(self, tag=tag)

    def explain_fields(self):
        return {"tag": self.tag}


class RenameTag:
    """Rename one tag value to another, one node at a time."""

    def __init__(self, old, new):
        self.name = f"rename_{old}_{new}"
        self.old, self.new = old, new

    def rewrite(self, root, context):
        def visit(node):
            children = tuple(visit(child) for child in node.children())
            if children != node.children():
                node = node.with_children(children)
            if isinstance(node, TaggedInput) and node.tag == self.old:
                return replace(node, tag=self.new)
            return node

        rewritten = visit(root)
        return None if rewritten is root else rewritten


def _tagged_plan(tag):
    field = ColumnRef("r", "reviews", "review")
    scan = Scan("reviews", "r", "review")
    return LogicalPlan(Project(TaggedInput(scan, tag), (field,)))


REWRITES = [
    ("old", [("old", "new")], "new", ("rename_old_new",)),
    # the second rule only matches after the first has run, and it is
    # listed first, so a single round would miss it
    ("old", [("mid", "new"), ("old", "mid")], "new",
     ("rename_old_mid", "rename_mid_new")),
    ("new", [("old", "new")], "new", ()),
]


def test_custom_logical_node_rewrites_until_a_round_changes_nothing():
    for start, rules, tag, changed in REWRITES:
        case = f"{start} {rules}"
        plan = _tagged_plan(start)
        optimized, applied = apply_logical_rules(
            plan, tuple(RenameTag(old, new) for old, new in rules), CONTEXT)

        assert [node.type_name for node in optimized.walk()] == [
            "quail.scan", "test.tagged_input", "quail.logical_project"], case
        assert optimized.root.input.tag == tag, case
        assert applied == changed, case
        assert (optimized.root is plan.root) == (not changed), case


def test_rules_that_never_settle_stop_at_the_pass_cap():
    # each round ends on a different tag than it started, so the rounds
    # would repeat forever: a -> b -> c | c -> a | a -> b -> c | ...
    rules = (RenameTag("a", "b"), RenameTag("c", "a"), RenameTag("b", "c"))

    optimized, changed = apply_logical_rules(
        _tagged_plan("a"), rules, CONTEXT, max_passes=3)

    assert optimized.root.input.tag == "c"
    assert len(changed) == 5
