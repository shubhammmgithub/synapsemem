"""Rules engine for relationship management in the graph"""

"""
Relationship rule definitions for the knowledge graph.

This module will contain symbolic reasoning rules
for graph-based inference in future versions of SynapseMem.

Examples for future versions:
- contradiction detection
- inference rules
- semantic relationship expansion
"""

RELATIONSHIP_RULES = {
    "likes": {
        "inverse": "liked_by",
        "type": "preference"
    },
    "works_on": {
        "inverse": "worked_by",
        "type": "project"
    },
    "lives_in": {
        "inverse": "location_of",
        "type": "location"
    },
}