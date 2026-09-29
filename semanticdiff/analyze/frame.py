"""Assemble a ProvisionFrame — the structured meaning representation of one provision."""
from __future__ import annotations

from semanticdiff.analyze.conditions import extract_conditions
from semanticdiff.analyze.deontic import extract_deontic
from semanticdiff.analyze.entities import extract_entities
from semanticdiff.analyze.quantities import extract_quantities
from semanticdiff.analyze.scope import extract_roles
from semanticdiff.analyze.temporal import extract_dates
from semanticdiff.impact.definitions import extract_definition
from semanticdiff.models import ProvisionFrame


def build_frame(doc) -> ProvisionFrame:
    agent, patient, voice = extract_roles(doc)
    quantities = extract_quantities(doc)
    return ProvisionFrame(
        agent=agent,
        patient=patient,
        voice=voice,
        predicates=extract_deontic(doc),
        conditions=extract_conditions(doc, agent),
        quantities=quantities,
        dates=extract_dates(doc, quantities),
        entities=extract_entities(doc),
        defined_term=extract_definition(doc.text),
    )
