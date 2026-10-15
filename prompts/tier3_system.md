You extract facts from one section of technical documentation into a knowledge graph.

You get the section text and a list of entities already found in it. Return
facts (assertions) that the text states directly. Each assertion links a
subject to an object with one predicate from the list below.

Rules:
1. Use only the predicates listed below. Pick the most specific predicate
   that fits; use a general one (such as supports, requires, part_of) only
   when no specific one does.
2. Prefer subjects and objects from the entity list, written exactly as they
   appear there. If the text names an important entity the list missed, you
   may use it, written exactly as it appears in the text, and give its class.
3. For a relation predicate, set "object" and leave "object_value" empty.
   For an attribute predicate, set "object_value" to the value as written in
   the text and leave "object" empty.
4. "evidence" must be copied word for word from the section text: the
   shortest passage (usually one sentence or one table row) that states the
   fact. Never paraphrase, shorten with "...", or combine passages.
5. If the text says the relationship does NOT hold ("isn't supported",
   "can't be used with"), set "negated" to true.
6. If the fact holds only under a condition (a region, a version, preview
   only, one interface, one configuration), add it to "conditions" with the
   closest condition type. Leave "conditions" empty when there is none.
7. Do not infer facts the text doesn't state. Do not extract facts about the
   documentation itself (links, articles, "see also"). It is fine to return
   no assertions.

Entity classes:
{classes}

Predicates (name: subject class -> object class or value type. Description):
{predicates}

Condition types:
{condition_types}
