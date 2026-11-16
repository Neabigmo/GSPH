"""Typed interaction patches for GSPH6.

An interaction patch keeps every directed atom-level microcontact.  The
nearest pair is only used for deterministic ordering; it is never used as a
replacement for the patch geometry or physical attributes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MicroContact:
    donor: int
    acceptor: int
    interaction_type: str
    distance: float
    tau: float
    attributes: Mapping[str, float] = field(default_factory=dict)

    @property
    def atom_ids(self) -> Tuple[int, int]:
        return int(self.donor), int(self.acceptor)


@dataclass
class InteractionPatch:
    patch_id: int
    donor_site: Tuple[str, int, str]
    acceptor_site: Tuple[str, int, str]
    contacts: List[MicroContact]
    donor_members: Tuple[int, ...]
    acceptor_members: Tuple[int, ...]
    geometry: Dict[str, float]
    physical: Dict[str, float] = field(default_factory=dict)

    @property
    def contact_count(self) -> int:
        return len(self.contacts)

    @property
    def atom_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(set(self.donor_members) | set(self.acceptor_members)))

    @property
    def minimum_tau(self) -> float:
        return min((float(c.tau) for c in self.contacts), default=0.0)


# Standard 20 amino acids (plus His protonation states).  Side-chain atoms
# get a chemical-group identity; backbone atoms are handled first.
_PROTEIN_GROUPS = {
    "ARG": "guanidinium", "LYS": "amine",
    "ASP": "carboxylate", "GLU": "carboxylate",
    "ASN": "amide", "GLN": "amide",
    "SER": "hydroxyl", "THR": "hydroxyl", "TYR": "phenol",
    "CYS": "thiol", "MET": "thioether",
    "HIS": "imidazole", "HID": "imidazole", "HIE": "imidazole",
    "HIP": "imidazole",
    "TRP": "indole", "PHE": "phenyl",
    "ALA": "methyl", "VAL": "isopropyl", "LEU": "isobutyl",
    "ILE": "secbutyl", "PRO": "pyrrolidine", "GLY": "backbone_only",
}
_PROTEIN_BACKBONE_N = {"N", "H", "HN", "H1", "H2", "H3", "HT1", "HT2", "HT3"}
_PROTEIN_BACKBONE_O = {"O", "OXT", "OT1", "OT2"}
_PROTEIN_BACKBONE_C = {"C", "CA"}
_PROTEIN_RESIDUE_NAMES = set(_PROTEIN_GROUPS)


def _functional_group_id(atom, role: str, base: str) -> str:
    """Return a conservative functional-group identity for one atom.

    Atoms within one chemical group share a key, while chemically distinct
    groups in the same residue remain separate.  This is intentionally a
    template-level identity; it does not invent FR3D/DSSR annotations.
    """
    name = str(getattr(atom, "name", "")).upper().replace("*", "'")
    res = str(getattr(atom, "resname", "")).upper()
    if res in _PROTEIN_RESIDUE_NAMES:
        # Backbone atoms share one site per residue face; side-chain atoms
        # share the residue's chemical group.  ASN/GLN and the previously
        # missing residues now resolve to amide/alkyl/etc. instead of being
        # silently classified as sugar/base atoms.
        if name in _PROTEIN_BACKBONE_N:
            return "backbone_N"
        if name in _PROTEIN_BACKBONE_O:
            return "backbone_carbonyl_O"
        if name in _PROTEIN_BACKBONE_C:
            return "backbone"
        return f"sidechain_{_PROTEIN_GROUPS[res]}"
    # Protein-like residues not in the table (nonstandard AAs) are still
    # protein atoms: they must NOT be labelled sugar/base groups.
    from cspph.core.gsph1 import _residue_class
    if _residue_class(res) != "nucleic" and res not in {"DA", "DC", "DG", "DT", "DI",
                                                       "RA", "RC", "RG", "RU", "RI",
                                                       "A", "C", "G", "U", "T", "I",
                                                       "ADE", "CYT", "GUA", "URA",
                                                       "THY", "PSU", "1MA", "5MC",
                                                       "OMG", "M2G", "7MG", "OMC"}:
        if name in _PROTEIN_BACKBONE_N:
            return "backbone_N"
        if name in _PROTEIN_BACKBONE_O:
            return "backbone_carbonyl_O"
        if name in _PROTEIN_BACKBONE_C:
            return "backbone"
        return f"residue_{res}"
    if name == "P" or name.startswith("OP") or name.startswith("O1P") or name.startswith("O2P"):
        return "phosphate"
    if name in {"O2'", "O2*"}:
        return "sugar_2prime_OH"
    if name.startswith("C") or name.startswith("O4'") or name.startswith("O4*"):
        return "sugar"
    return f"base_{base}_{_base_edge(name, base)}"


# Approximate Leontis-Westhof base-edge assignment for the heteroatoms that
# carry donor/acceptor roles.  Atoms not listed keep the generic edge label;
# the sugar edge also absorbs backbone/sugar-facing base atoms.  This is a
# template-level approximation, not an FR3D/DSSR annotation.
_BASE_EDGES = {
    "A":  {"N1": "wc", "N6": "wc", "N7": "hoogsteen"},
    "G":  {"N1": "wc", "N2": "wc", "O6": "wc", "N3": "sugar",
           "N7": "hoogsteen"},
    "C":  {"N3": "wc", "N4": "wc", "O2": "wc"},
    "U":  {"N3": "wc", "O2": "wc", "O4": "wc"},
    "T":  {"N3": "wc", "O2": "wc", "O4": "wc"},
    "I":  {"N1": "wc", "N2": "wc", "O6": "wc", "N3": "sugar",
           "N7": "hoogsteen"},
}


def _base_edge(name: str, base: str) -> str:
    return _BASE_EDGES.get(str(base).upper(), {}).get(str(name).upper(), "edge")


def _site_key(atom, role: str, base: str) -> Tuple[str, int, str]:
    """Stable functional-site key; residue + insertion-code identity kept."""
    icode = str(getattr(atom, "icode", "") or "")
    suffix = f"{icode}>{role}:{_functional_group_id(atom, role, base)}" if icode else f"{role}:{_functional_group_id(atom, role, base)}"
    return (str(atom.chain), int(atom.resid), suffix)


def _geometry(contacts: Sequence[MicroContact], atoms: Mapping[int, object]) -> Dict[str, float]:
    ids = sorted({c.donor for c in contacts} | {c.acceptor for c in contacts})
    xyz = np.asarray([np.asarray(atoms[i].xyz, dtype=float) for i in ids], dtype=float)
    if xyz.size == 0:
        return {"center_x": 0.0, "center_y": 0.0, "center_z": 0.0,
                "extent": 0.0, "anisotropy": 0.0}
    center = xyz.mean(axis=0)
    cov = np.cov(xyz - center, rowvar=False, bias=True) if len(xyz) > 1 else np.zeros((3, 3))
    vals = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0.0))[::-1]
    return {
        "center_x": float(center[0]), "center_y": float(center[1]), "center_z": float(center[2]),
        "extent": float(np.max(np.linalg.norm(xyz - center, axis=1))) if len(xyz) else 0.0,
        "anisotropy": float((vals[0] - vals[-1]) / max(vals[0], 1e-12)) if vals.size else 0.0,
        "atom_count": float(len(ids)),
    }


def build_interaction_patches(events: Sequence[object], atoms: Sequence[object], patch_catalog=None) -> List[InteractionPatch]:
    """Aggregate directed events into complete chemical patches.

    Events are grouped by donor/acceptor functional sites, not by distance.
    Every event remains a :class:`MicroContact`, preserving all local physical
    information and the direction of ordered RDF/contact channels.
    """
    atom_map = {int(a.index): a for a in atoms}
    groups: Dict[Tuple[Tuple[str, int, str], Tuple[str, int, str]], List[MicroContact]] = {}
    for event in events:
        donor = atom_map[int(event.donor)]; acceptor = atom_map[int(event.acceptor)]
        # The event direction/base annotation is candidate-local.  A stable
        # patch identity must depend only on atom identity and residue/base
        # chemistry, otherwise a rigid decoy can renumber every downstream
        # path.  _base_class is imported lazily to avoid a gsph1->patches
        # import cycle at module import time.
        from cspph.core.gsph1 import _base_class
        donor_site = _site_key(donor, "donor", _base_class(str(getattr(donor, "resname", ""))))
        acceptor_site = _site_key(acceptor, "acceptor", _base_class(str(getattr(acceptor, "resname", ""))))
        # A patch is a functional-site pair.  Interaction type is retained on
        # each MicroContact, but must not split one chemical site into several
        # topological vertices.
        key = (donor_site, acceptor_site)
        groups.setdefault(key, []).append(MicroContact(
            donor=int(event.donor), acceptor=int(event.acceptor),
            interaction_type=str(event.event_type), distance=float(event.distance),
            tau=float(event.tau), attributes=dict(event.attributes),
        ))
    patches: List[InteractionPatch] = []
    for key in sorted(groups, key=lambda x: (x[0], x[1])):
        patch_id = int(patch_catalog.id_for_key(key)) if patch_catalog is not None else len(patches)
        contacts = sorted(groups[key], key=lambda c: (c.tau, c.distance, c.donor, c.acceptor))
        donors = tuple(sorted({c.donor for c in contacts})); acceptors = tuple(sorted({c.acceptor for c in contacts}))
        patches.append(InteractionPatch(
            patch_id=patch_id, donor_site=key[0], acceptor_site=key[1], contacts=contacts,
            donor_members=donors, acceptor_members=acceptors,
            geometry=_geometry(contacts, atom_map),
        ))
    return patches


__all__ = ["MicroContact", "InteractionPatch", "build_interaction_patches"]
