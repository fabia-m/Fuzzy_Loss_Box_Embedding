"""
Fuzzy Logic-Based Loss Functions for Box Embeddings
Chapter 3: Curriculum Validation

This module provides:
- Ontology loading utilities (OWL parsing, noise injection)
- Fuzzy box embedding model with configurable t-norms/implications
- Curriculum‑based training with phased loss activation
- Evaluation metrics (satisfaction degrees, violations)
"""
# pip install --upgrade pip
# pip install rdflib pykeen pandas owlready2 torch owlrl matplotlib numpy seaborn

# Standard library imports
import json
import math
import os
import pickle
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Set

# Third‑party imports
import numpy as np
import pandas as pd
import rdflib
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdflib.collection import Collection
from rdflib.namespace import RDF, RDFS, OWL
import hashlib

# Set default torch dtype
torch.set_default_dtype(torch.float32)

def _uri_str(node: rdflib.term.Node) -> Optional[str]:
    """Extract URI string from RDF node."""
    return str(node) if isinstance(node, rdflib.term.URIRef) else None

def _ordered_pair(a: str, b: str) -> Tuple[str, str]:
    """Return ordered pair for consistent disjoint edge representation."""
    return (a, b) if a < b else (b, a)

def _local_name(uri: str) -> str:
    """Extract local name from URI (after # or /)."""
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rsplit("/", 1)[-1]

def load_owl(
    owl_path: str,
    allowed_namespaces: Optional[Set[str]] = None,
) -> Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Extract named classes, subclass relations, and disjoint pairs from an OWL ontology file.

    Args:
        owl_path: Path to OWL file.
        allowed_namespaces: Optional filter for class URIs.

    Returns:
        classes: List of class URIs.
        subclass_of: List of (child, parent) tuples.
        disjoint_pairs: List of (class_a, class_b) tuples.
    """
    g = rdflib.Graph()
    g.parse(owl_path, format="xml")

    def _keep(uri: str) -> bool:
        if allowed_namespaces is None:
            return True
        return any(uri.startswith(ns) for ns in allowed_namespaces)

    classes: Set[str] = set()

    # Collect all OWL classes
    for c in g.subjects(RDF.type, OWL.Class):
        uri = _uri_str(c)
        if uri and _keep(uri):
            classes.add(uri)

    # Ensure both child and parent URIs are recorded
    for child, _, parent in g.triples((None, RDFS.subClassOf, None)):
        child_uri = _uri_str(child)
        parent_uri = _uri_str(parent)
        if child_uri and _keep(child_uri):
            classes.add(child_uri)
        if parent_uri and _keep(parent_uri):
            classes.add(parent_uri)

    # Subclass relations (including equivalent classes)
    subclass_of: Set[Tuple[str, str]] = set()
    for child, _, parent in g.triples((None, RDFS.subClassOf, None)):
        child_uri, parent_uri = _uri_str(child), _uri_str(parent)
        if child_uri and parent_uri and _keep(child_uri) and _keep(parent_uri):
            subclass_of.add((child_uri, parent_uri))
    for a, _, b in g.triples((None, OWL.equivalentClass, None)):
        a_uri, b_uri = _uri_str(a), _uri_str(b)
        if a_uri and b_uri and _keep(a_uri) and _keep(b_uri):
            subclass_of.add((a_uri, b_uri))
            subclass_of.add((b_uri, a_uri))

    # Disjoint pairs (including AllDisjointClasses)
    disjoint_pairs: Set[Tuple[str, str]] = set()
    for a, _, b in g.triples((None, OWL.disjointWith, None)):
        a_uri, b_uri = _uri_str(a), _uri_str(b)
        if a_uri and b_uri and _keep(a_uri) and _keep(b_uri):
            disjoint_pairs.add(_ordered_pair(a_uri, b_uri))
    for adc in g.subjects(RDF.type, OWL.AllDisjointClasses):
        for _, _, members in g.triples((adc, OWL.members, None)):
            try:
                collection = Collection(g, members)
            except Exception:
                continue
            member_uris = [uri for m in collection if (uri := _uri_str(m)) and _keep(uri)]
            for i in range(len(member_uris)):
                for j in range(i + 1, len(member_uris)):
                    disjoint_pairs.add(_ordered_pair(member_uris[i], member_uris[j]))

    classes_list = sorted(classes)
    subclass_list = sorted(subclass_of)
    disjoint_list = sorted(disjoint_pairs)

    print(
        f"Loaded {owl_path}: {len(classes_list)} classes, "
        f"{len(subclass_list)} subclass axioms, {len(disjoint_list)} disjoint pairs"
    )
    return classes_list, subclass_list, disjoint_list

class OntologyEdges:
    """Stores subclass and disjoint edges for an ontology.

    Computes closure, entailments, depths, and sibling relationships.
    Provides fuzzy loss helpers for training.
    """

    def __init__(
        self,
        classes: List[str],
        subclass_of: List[Tuple[str, str]],
        disjoint_pairs: List[Tuple[str, str]],
        device: Optional[str] = None,
    ):
        self.device = device or "mps"
        self.cls2id = {uri: i for i, uri in enumerate(classes)}
        self.id2cls = {i: uri for i, uri in enumerate(classes)}
        self.num_classes = len(classes)

        # Subclass edges
        self.asserted_sub_edges = [(self.cls2id[c], self.cls2id[p]) for c, p in subclass_of]
        self.closure_sub_edges_with_depth = self._compute_closure_with_depth(self.asserted_sub_edges)
        self.closure_sub_edges = torch.tensor(
            [(c, a) for c, a, _ in self.closure_sub_edges_with_depth],
            dtype=torch.long,
            device=self.device,
        )
        self.closure_depths = torch.tensor(
            [d for _, _, d in self.closure_sub_edges_with_depth],
            dtype=torch.float,
            device=self.device,
        )

        # Disjoint edges
        self.asserted_disjoint_edges = [(self.cls2id[a], self.cls2id[b]) for a, b in disjoint_pairs]
        entailed_disjoint = self._entail_disjointness(
            self.asserted_disjoint_edges, self.closure_sub_edges_with_depth
        )
        self.entailed_disjoint_edges = torch.tensor(
            list(entailed_disjoint), dtype=torch.long, device=self.device
        )

        # Per‑edge entailment weight
        self.entail_count = self._compute_entail_count()

        # Per‑class depth from root (used in oversize loss)
        depths = torch.full((self.num_classes,), float("inf"), device=self.device)
        all_children = {c for c, _ in self.asserted_sub_edges}
        all_parents = {p for _, p in self.asserted_sub_edges}
        for r in all_parents - all_children:
            depths[r] = 0.0
        changed = True
        while changed:
            changed = False
            for c, p in self.asserted_sub_edges:
                if depths[p] + 1 < depths[c]:
                    depths[c] = depths[p] + 1
                    changed = True
        depths[torch.isinf(depths)] = depths[~torch.isinf(depths)].max()
        self.class_depths = depths

        # Sibling edges
        sibling_pairs = self._compute_sibling_pairs()
        self.sibling_edges = (
            torch.tensor(sibling_pairs, dtype=torch.long, device=self.device)
            if sibling_pairs
            else None
        )

        # Violation tracking (for compatibility with old API)
        self.last_subclass_violations: Optional[int] = None
        self.last_disjoint_violations: Optional[int] = None
        self.last_sibling_distance: Optional[float] = None

    # Internal helpers
    def _compute_closure_with_depth(self, edges):
        parents = {i: {} for i in range(self.num_classes)}
        for c, p in edges:
            parents[c][p] = 1
        changed = True
        while changed:
            changed = False
            for c in range(self.num_classes):
                for p, d in list(parents[c].items()):
                    for gp, gd in parents[p].items():
                        new_d = d + gd
                        if gp not in parents[c] or new_d < parents[c][gp]:
                            parents[c][gp] = new_d
                            changed = True
        # Filter self-edges (c == a): equivalence cycles produce (a, a, depth)
        # self-loops which are permanently unsatisfiable under Reichenbach
        # implication (boundary membership 0.5 -> I(0.5,0.5)=0.75 < alpha).
        return [
            (c, a, d) for c, anc in parents.items()
            for a, d in anc.items() if c != a
        ]

    def _entail_disjointness(self, disjoint_edges, subclass_closure) -> set:
        descendants = {}
        for c, a, _ in subclass_closure:
            descendants.setdefault(a, set()).add(c)
        entailed = set(disjoint_edges)
        for a, b in disjoint_edges:
            for da in descendants.get(a, []):
                entailed.add(tuple(sorted((da, b))))
            for db in descendants.get(b, []):
                entailed.add(tuple(sorted((a, db))))
        return entailed

    def _compute_entail_count(self):
        descendants = defaultdict(set)
        ancestors = defaultdict(set)
        for c, p, _ in self.closure_sub_edges_with_depth:
            descendants[p].add(c)
            ancestors[c].add(p)
        for i in range(self.num_classes):
            descendants[i].add(i)
            ancestors[i].add(i)
        counts = [
            len(descendants[c]) * len(ancestors[p])
            for c, p, _ in self.closure_sub_edges_with_depth
        ]
        return torch.log1p(
            torch.tensor(counts, dtype=torch.float, device=self.device)
        )

    def _compute_sibling_pairs(self, max_pairs_per_parent: Optional[int] = 200, seed: int = 0) -> List[Tuple[int, int]]:
        rng = random.Random(seed)
        parent_to_children: dict = defaultdict(set)
        for child, parent in self.asserted_sub_edges:
            parent_to_children[parent].add(child)
        pairs: set = set()
        for children in parent_to_children.values():
            children = list(children)
            n = len(children)
            all_pairs_count = n * (n - 1) // 2
            if max_pairs_per_parent is None or all_pairs_count <= max_pairs_per_parent:
                for i in range(n):
                    for j in range(i + 1, n):
                        pairs.add(tuple(sorted((children[i], children[j]))))
            else:
                sampled = 0
                attempts = 0
                max_attempts = max_pairs_per_parent * 20
                while sampled < max_pairs_per_parent and attempts < max_attempts:
                    a, b = rng.sample(children, 2)
                    p = tuple(sorted((a, b)))
                    if p not in pairs:
                        pairs.add(p)
                        sampled += 1
                    attempts += 1
        return list(pairs)

    def check_entailed_and_closure_edges(
        self, mn: torch.Tensor, mx: torch.Tensor, disjoint_margin: float = 0.02
    ) -> dict:
        """Check entailed disjoint and transitive closure subclass edges against current embedding."""
        result = {}
        # Closure subclass edges
        if self.closure_sub_edges.numel() > 0:
            child = self.closure_sub_edges[:, 0]
            parent = self.closure_sub_edges[:, 1]
            lower_ok = (mn[parent] <= mn[child])
            upper_ok = (mx[child] <= mx[parent])
            edge_ok = (lower_ok & upper_ok).all(dim=1)
            total = edge_ok.numel()
            found = edge_ok.sum().item()
        else:
            total = found = 0
        result["closure_sub_total"] = total
        result["closure_sub_found"] = int(found)
        result["closure_sub_rate"] = found / total if total > 0 else 1.0

        # Entailed disjoint edges
        if self.entailed_disjoint_edges.numel() > 0:
            a = self.entailed_disjoint_edges[:, 0]
            b = self.entailed_disjoint_edges[:, 1]
            sep_ab = mn[b] - mx[a]
            sep_ba = mn[a] - mx[b]
            sep = torch.maximum(sep_ab, sep_ba)
            pair_ok = (sep.max(dim=1).values >= disjoint_margin)
            dis_total = pair_ok.numel()
            dis_found = pair_ok.sum().item()
        else:
            dis_total = dis_found = 0
        result["entailed_dis_total"] = dis_total
        result["entailed_dis_found"] = int(dis_found)
        result["entailed_dis_rate"] = dis_found / dis_total if dis_total > 0 else 1.0
        return result

    # Fuzzy loss methods (replace penalty based ones)
    def subclass_loss(self, model: 'FuzzyBoxEmbedding', batch_size: int = 512) -> torch.Tensor:
        """Fuzzy subclass loss using containment degree.
        
        Processes edges in batches to bound memory usage.
        
        Args:
            model: Fuzzy box embedding model
            batch_size: Number of edges to process at once (default 512)
        """
        n_edges = self.closure_sub_edges.shape[0]
        if n_edges == 0:
            return torch.tensor(0.0, device=model.center.device)
        
        losses = []
        
        for i in range(0, n_edges, batch_size):
            end = min(i + batch_size, n_edges)
            child_idx = self.closure_sub_edges[i:end, 0]
            parent_idx = self.closure_sub_edges[i:end, 1]
            
            contain_degrees = model.containment_degree(child_idx, parent_idx)
            per_edge_loss = 1.0 - contain_degrees
            
            if hasattr(self, 'entail_count') and self.entail_count is not None:
                weighted = self.entail_count[i:end] * per_edge_loss
                losses.append(weighted)
            else:
                losses.append(per_edge_loss)
        
        # Concatenate all batch losses and compute mean
        all_losses = torch.cat(losses, dim=0)
        return all_losses.mean()

    def disjoint_loss(self, model: 'FuzzyBoxEmbedding', cfg: Optional['FuzzyBoxConfig'] = None,
                      use_entailed: bool = True, batch_size: int = 512) -> torch.Tensor:
        """Fuzzy disjoint loss using overlap degree.

        Processes edges in batches to bound memory usage.

        When ``cfg.contact_weight > 0``, a crisp margin-tail term is added
        (penalty-style arbitration): ``contact_weight *
        softplus(margin - sep)`` with ``sep`` the max-over-dims crisp
        separation and ``margin = 0.02`` matching the evaluation margin.
        The fuzzy overlap tail dies at ~3*tau, so the soft overlap gradient
        alone cannot hold asserted disjoint pairs apart — the tail is
        LOAD-BEARING, not a legacy of the sibling term: ablation P1a
        (2026-10-07, no-sibling config, probe/nocontact/) shows
        contact 0 vs 0.5 on OEO d2 = disjoint 35.06% vs 1.49% mean over
        seeds {42,4,666,1010} (DOID d2 fine either way: 0.41% vs 0.00%).
        Keep ``contact_weight=0.5`` (the dataclass default).

        Args:
            model: Fuzzy box embedding model
            cfg: Optional config carrying ``contact_weight``
            use_entailed: Whether to use entailed disjoint edges
            batch_size: Number of edges to process at once (default 512)
        """
        dis_edges = (
            self.entailed_disjoint_edges
            if use_entailed
            else torch.tensor(self.asserted_disjoint_edges, dtype=torch.long, device=self.device)
        )
        
        n_edges = dis_edges.shape[0]
        if n_edges == 0:
            return torch.tensor(0.0, device=model.center.device)
        
        overlaps = []
        
        for i in range(0, n_edges, batch_size):
            end = min(i + batch_size, n_edges)
            a = dis_edges[i:end, 0]
            b = dis_edges[i:end, 1]
            
            overlap_degrees = model.overlap_degree(a, b)
            overlaps.append(overlap_degrees)
        
        # Concatenate all batch overlaps and compute mean
        all_overlaps = torch.cat(overlaps, dim=0)
        loss = all_overlaps.mean()

        if cfg is not None and cfg.contact_weight > 0:
            margin = 0.02  # evaluation margin (matches check_disjoint + penalty baseline)
            mn, mx = model.get_min_max()
            # Contact tail acts on ASSERTED disjoint pairs only: the entailed
            # set would multiply the shove through descendant boxes (~6 pairs
            # per box on OEO) and break containment.
            if getattr(cfg, "contact_on_entailed", False):
                contact_src = self.entailed_disjoint_edges
            else:
                contact_src = torch.tensor(self.asserted_disjoint_edges, dtype=torch.long,
                                           device=self.device)
            n_contact = contact_src.shape[0]
            seps = []
            for i in range(0, n_contact, batch_size):
                end = min(i + batch_size, n_contact)
                a = contact_src[i:end, 0]
                b = contact_src[i:end, 1]
                sep = torch.max(torch.max(mn[b] - mx[a], mn[a] - mx[b]), dim=1).values
                seps.append(sep)
            all_seps = torch.cat(seps, dim=0)
            loss = loss + cfg.contact_weight * F.softplus(margin - all_seps).mean()
        return loss

    def min_box_loss(self, model: 'FuzzyBoxEmbedding', cfg: 'FuzzyBoxConfig') -> torch.Tensor:
        """Floor penalty: push per-dimension box widths above min_box_size.

        Mirrors the penalty-based size loss (`size_weight * softplus(min_box_size - (mx-mn))`),
        adapted to fuzzy half-sizes (width = 2*delta per dimension).
        """
        widths = 2.0 * model.half_size()
        return F.softplus(cfg.min_box_size - widths.clamp(min=1e-6)).mean()

    def oversize_loss(self, model: 'FuzzyBoxEmbedding', cfg: 'FuzzyBoxConfig') -> torch.Tensor:
        """Fuzzy oversize loss based on log cardinality.

        Two-sided, mirrors the penalty-based oversize loss: penalises both
        overshooting the depth-scaled target AND undershooting min_log_volume
        (prevents box collapse). Uses fuzzy cardinality instead of crisp volumes.
        """
        log_card = model.fuzzy_cardinality()
        depths = self.class_depths
        if len(depths) != model.n_classes:
            depths = torch.zeros(model.n_classes, device=log_card.device)
        target = cfg.base_log_volume - cfg.depth_scale * torch.sqrt(depths)
        target = torch.clamp(target, min=cfg.min_log_volume)
        return (
            F.softplus(log_card - target) +
            F.softplus(cfg.min_log_volume - log_card)
        ).mean()

    def distance_loss(self, model: 'FuzzyBoxEmbedding', batch_size: int = 512) -> torch.Tensor:
        """Fuzzy sibling distance loss (penalise gaps between sibling boxes).

        Pulls siblings together, removed
        
        """
        if self.sibling_edges is None or self.sibling_edges.numel() == 0:
            return torch.tensor(0.0, device=model.center.device)

        losses = []
        for i in range(0, self.sibling_edges.shape[0], batch_size):
            end = min(i + batch_size, self.sibling_edges.shape[0])
            a = self.sibling_edges[i:end, 0]
            b = self.sibling_edges[i:end, 1]

            # Graded gap: 1 = separated, 0 = fully overlapping
            losses.append(1.0 - model.overlap_degree(a, b))

        return torch.cat(losses, dim=0).mean()

@dataclass
class FuzzyBoxConfig:
    """Configuration for fuzzy box embeddings. Same as for penalty models."""
    dim: int = 6
    steps: int = 5000  # Reduced from 10k — plateau reached by 5k
    lr: float = 0.05
    seed: int = 42
    membership_type: str = "sigmoidal"
    t_norm: str = "product"
    implication: str = "reichenbach"
    learn_temperature: bool = False
    init_temperature: float = 0.1 # FIXED
    min_box_size: float = 0.05
    size_weight: float = 0.1
    subclass_weight: float = 2.0
    disjoint_weight: float = 2.0  # soft overlap penalty needs higher disjoint pressure than margin-based
    contact_weight: float = 0.5  # crisp margin-tail on disjointness
    contact_on_entailed: bool = False
    big_box_weight: float = 0.1
    distance_weight: float = 0.0  # sibling term DROPPED
    base_log_volume: float = 2.0
    depth_scale: float = 0.5
    min_log_volume: float = -4.0
    
    def __post_init__(self):
        pass  # lr is fixed at 0.05
def cfg_run_name(cfg: 'FuzzyBoxConfig') -> str:
    """Uniform run-name for save dirs: {t_norm}_{implication}_{learn|fix}tau{temp}.
    """
    suffix = "learn" if cfg.learn_temperature else "fix"
    return f"{cfg.t_norm}_{cfg.implication}"


def run_dir_name(cfg: 'FuzzyBoxConfig', seed: int) -> str:
    """Uniform result dir name: base run name for the default seed (42),
    suffixed with _seed{N} for any other seed. """
    base = cfg_run_name(cfg)
    return base if seed == 42 else f"{base}_seed{seed}"


def is_done(onto: str, seed: int, dim: int, results_dir: str = "results",
            cfg: Optional['FuzzyBoxConfig'] = None) -> bool:
    """True if results/{onto}/{run_dir_name}/dim_{N} already has a valid run.

    Valid = metrics.json exists with a real (non-None) train_time_seconds AND
    config.json matches the current weights (sub 2.0 / disj 2.0 / contact 0.5 / no sibling). A run with a
    None timestamp or stale weights is
    treated as not-done so a resume re-trains it."""
    from pathlib import Path as _Path
    if cfg is None:
        cfg = FuzzyBoxConfig()
    d = _Path(results_dir) / onto / run_dir_name(cfg, seed) / f"dim_{dim}"
    mf, cf = d / "metrics.json", d / "config.json"
    if not (mf.exists() and cf.exists()):
        return False
    try:
        m, c = json.loads(mf.read_text()), json.loads(cf.read_text())
    except (json.JSONDecodeError, OSError, ValueError):
        return False
    if m.get("train_time_seconds") is None:
        return False
    return (c.get("subclass_weight", 2.0) == (cfg.subclass_weight if cfg else 2.0) and
            c.get("disjoint_weight", 2.0) == (cfg.disjoint_weight if cfg else 2.0) and
            c.get("contact_weight", 0.5) == (cfg.contact_weight if cfg else 0.5) and
            c.get("contact_on_entailed", False) == (cfg.contact_on_entailed if cfg else False) and
            c.get("distance_weight", 0.0) == (cfg.distance_weight if cfg else 0.0) and
            c.get("big_box_weight", 0.1) == (cfg.big_box_weight if cfg else 0.1))


@dataclass
class CurriculumSchedule:
    """Phased activation schedule for curriculum learning."""
    subclass_start: float = 0.0
    disjoint_start: float = 0.01
    sibling_start: float = 0.5
    big_box_start: float = 0.7
    ramp: bool = False

class FuzzyBoxEmbedding(nn.Module):
    """Fuzzy box embedding model with configurable membership functions."""

    def __init__(self, n_classes: int, dim: int, cfg: FuzzyBoxConfig, seed: int = 0):
        super().__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        self.center = nn.Parameter(torch.randn(n_classes, dim, generator=g) * 0.1)
        self.raw_half_size = nn.Parameter(torch.randn(n_classes, dim, generator=g) * 0.1)
        if cfg.learn_temperature:
            self.raw_temperature = nn.Parameter(
                torch.full((n_classes, dim), math.log(cfg.init_temperature))
            )
        else:
            self.register_buffer("raw_temperature", None)
        self.membership_fn = FuzzyMembership.get(cfg.membership_type)
        self.t_norm = TNorms.get(cfg.t_norm)
        self.implication = Implications.get(cfg.implication)

    def half_size(self) -> torch.Tensor:
        return F.softplus(self.raw_half_size, beta=1.0) + 1e-6

    def temperature(self) -> torch.Tensor:
        if self.raw_temperature is None:
            return torch.full_like(self.half_size(), self.cfg.init_temperature)
        return torch.exp(self.raw_temperature).clamp(min=1e-4, max=10.0)

    def get_min_max(self) -> Tuple[torch.Tensor, torch.Tensor]:
        hs = self.half_size()
        return self.center - hs, self.center + hs

    def containment_degree(self, child_idx: torch.Tensor, parent_idx: torch.Tensor) -> torch.Tensor:
        """Compute degree to which child ⊆ parent using fuzzy implication.
        
        Tests membership at center + all boundary points (±half_size per dimension).
        Uses vectorized computation for memory efficiency.
        """
        child_center = self.center[child_idx]
        parent_center = self.center[parent_idx]
        child_hs = self.half_size()[child_idx]
        parent_hs = self.half_size()[parent_idx]
        
        use_temp = self.cfg.membership_type == "sigmoidal"
        child_temp = self.temperature()[child_idx] if use_temp else None
        parent_temp = self.temperature()[parent_idx] if use_temp else None
        
        n_edges = child_center.shape[0]
        n_points = 1 + 2 * self.dim
        
        # Build all test points at once: [n_edges, n_points, dim]
        # Point 0: center
        # Points 1,2: center ± hs on dim 0
        # Points 3,4: center ± hs on dim 1
        # etc.
        test_points = child_center.unsqueeze(1).expand(-1, n_points, -1).clone()
        
        for d in range(self.dim):
            test_points[:, 1 + 2*d, d] = child_center[:, d] + child_hs[:, d]
            test_points[:, 2 + 2*d, d] = child_center[:, d] - child_hs[:, d]
        
        # Reshape for batch evaluation: [n_edges * n_points, dim]
        test_points_flat = test_points.reshape(-1, self.dim)
        
        # Expand params for batch: [n_edges * n_points, dim]
        child_center_exp = child_center.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, self.dim)
        parent_center_exp = parent_center.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, self.dim)
        child_hs_exp = child_hs.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, self.dim)
        parent_hs_exp = parent_hs.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, self.dim)
        
        if use_temp and child_temp is not None and parent_temp is not None:
            child_temp_exp = child_temp.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, self.dim)
            parent_temp_exp = parent_temp.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, self.dim)
            mu_child = self.membership_fn(test_points_flat, child_center_exp, child_hs_exp, child_temp_exp)
            mu_parent = self.membership_fn(test_points_flat, parent_center_exp, parent_hs_exp, parent_temp_exp)
        else:
            mu_child = self.membership_fn(test_points_flat, child_center_exp, child_hs_exp)
            mu_parent = self.membership_fn(test_points_flat, parent_center_exp, parent_hs_exp)
        
        # Reshape back: [n_edges, n_points, dim]
        mu_child = mu_child.reshape(n_edges, n_points, self.dim)
        mu_parent = mu_parent.reshape(n_edges, n_points, self.dim)
        
        # Implication at each test point and dimension
        impl_degrees = self.implication(mu_child, mu_parent)
        
        # Minimum across all test points AND dimensions (sequential reduction)
        impl_degrees = impl_degrees.min(dim=2).values  # Min over dimensions
        return impl_degrees.min(dim=1).values  # Min over test points

    def overlap_degree(self, idx_a: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
        """Compute degree to which two boxes overlap using t‑norm."""
        center_a = self.center[idx_a]
        center_b = self.center[idx_b]
        hs_a = self.half_size()[idx_a]
        hs_b = self.half_size()[idx_b]
        use_temp = self.cfg.membership_type == "sigmoidal"
        temp_a = self.temperature()[idx_a] if use_temp else None
        temp_b = self.temperature()[idx_b] if use_temp else None
        
        # Test points: midpoint, center_a, center_b
        midpoint = (center_a + center_b) / 2
        n_edges = center_a.shape[0]
        n_points = 3
        
        # Stack test points: [n_edges, n_points, dim]
        test_points = torch.stack([midpoint, center_a, center_b], dim=1)
        
        # Flatten for batch evaluation: [n_edges * n_points, dim]
        test_points_flat = test_points.reshape(-1, self.dim)
        
        # Expand params: [n_edges * n_points, dim]
        center_a_exp = center_a.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, self.dim)
        center_b_exp = center_b.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, self.dim)
        hs_a_exp = hs_a.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, self.dim)
        hs_b_exp = hs_b.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, self.dim)
        
        if use_temp and temp_a is not None and temp_b is not None:
            temp_a_exp = temp_a.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, self.dim)
            temp_b_exp = temp_b.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, self.dim)
            mu_a = self.membership_fn(test_points_flat, center_a_exp, hs_a_exp, temp_a_exp)
            mu_b = self.membership_fn(test_points_flat, center_b_exp, hs_b_exp, temp_b_exp)
        else:
            mu_a = self.membership_fn(test_points_flat, center_a_exp, hs_a_exp)
            mu_b = self.membership_fn(test_points_flat, center_b_exp, hs_b_exp)
        
        # Reshape: [n_edges, n_points, dim]
        mu_a = mu_a.reshape(n_edges, n_points, self.dim)
        mu_b = mu_b.reshape(n_edges, n_points, self.dim)
        
        # T-norm at each point and dimension
        overlap = self.t_norm(mu_a, mu_b)
        
        # Max over test points, min over dimensions
        overlap = overlap.max(dim=1).values  # [n_edges, dim]
        return overlap.min(dim=1).values  # [n_edges]

    def fuzzy_cardinality(self) -> torch.Tensor:
        """Compute log fuzzy cardinality (integral of membership)."""
        hs = self.half_size()
        if self.cfg.membership_type == "sigmoidal":
            temp = self.temperature()
            integrals = 2.0 * hs + math.pi * temp
        else:
            integrals = hs
        return torch.log(integrals.clamp(min=1e-8)).sum(dim=-1)

    def volumes(self) -> torch.Tensor:
        return torch.log((2.0 * self.half_size()).clamp(min=1e-8)).sum(dim=-1)

class TNorms:
    """Fuzzy conjunction operators."""
    @staticmethod
    def minimum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.minimum(a, b)

    @staticmethod
    def product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b

    @staticmethod
    def lukasiewicz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.clamp(a + b - 1.0, min=0.0)

    @classmethod
    def get(cls, name: str):
        mapping = {
            "minimum": cls.minimum,
            "min": cls.minimum,
            "product": cls.product,
            "prod": cls.product,
            "lukasiewicz": cls.lukasiewicz,
            "luka": cls.lukasiewicz,
        }
        if name not in mapping:
            raise ValueError(f"Unknown t-norm: {name}")
        return mapping[name]

class Implications:
    """Fuzzy implications."""
    @staticmethod
    def kleene_dienes(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.maximum(1.0 - a, b)

    @staticmethod
    def reichenbach(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return 1.0 - a + a * b

    @staticmethod
    def lukasiewicz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.clamp(1.0 - a + b, max=1.0)

    @staticmethod
    def goguen(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.where(a <= b, torch.ones_like(a), b / (a + 1e-8))

    @staticmethod
    def godel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.where(a <= b, torch.ones_like(a), b)

    @classmethod
    def get(cls, name: str):
        mapping = {
            "kleene_dienes": cls.kleene_dienes,
            "kd": cls.kleene_dienes,
            "reichenbach": cls.reichenbach,
            "reic": cls.reichenbach,
            "lukasiewicz": cls.lukasiewicz,
            "luka": cls.lukasiewicz,
            "goguen": cls.goguen,
            "godel": cls.godel,
        }
        if name not in mapping:
            raise ValueError(f"Unknown implication: {name}")
        return mapping[name]

class FuzzyMembership:
    """Fuzzy membership functions for box embeddings."""
    @staticmethod
    def triangular(x: torch.Tensor, center: torch.Tensor, half_width: torch.Tensor) -> torch.Tensor:
        normalized_dist = torch.abs(x - center) / half_width.clamp(min=1e-8)
        return torch.clamp(1.0 - normalized_dist, min=0.0)

    @staticmethod
    def sigmoidal(x: torch.Tensor, center: torch.Tensor, half_width: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        temp = temperature.clamp(min=1e-4, max=10.0)
        margin = half_width - torch.abs(x - center)
        return torch.sigmoid(margin / temp)

    @staticmethod
    def gaussian(x: torch.Tensor, center: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        std = std.clamp(min=1e-4)
        diff = x - center
        return torch.exp(-0.5 * (diff / std) ** 2)

    @classmethod
    def get(cls, name: str):
        mapping = {
            "triangular": cls.triangular,
            "triangle": cls.triangular,
            "sigmoidal": cls.sigmoidal,
            "sigmoid": cls.sigmoidal,
            "gaussian": cls.gaussian,
            "gauss": cls.gaussian,
        }
        if name not in mapping:
            raise ValueError(f"Unknown membership: {name}")
        return mapping[name]


# ── Predefined curriculum schedules ────────────────────────────────────────
# Reusable schedule configurations for common experimental conditions.

SCHEDULES = {
    "plain": CurriculumSchedule(
        subclass_start=0.0,
        disjoint_start=0.0,
        sibling_start=0.0,
        big_box_start=0.0,
        ramp=False,
    ),
    "early": CurriculumSchedule(
        subclass_start=0.0,
        disjoint_start=0.01,  # 1% — Ch. 2 optimal
        sibling_start=0.5,
        big_box_start=0.7,
        ramp=False,
    ),
    "mid": CurriculumSchedule(
        subclass_start=0.0,
        disjoint_start=0.1,  # 10% — wait for containment
        sibling_start=0.5,
        big_box_start=0.7,
        ramp=False,
    ),
    "late": CurriculumSchedule(
        subclass_start=0.0,
        disjoint_start=0.7,  # 70% — late activation
        sibling_start=0.0,
        big_box_start=0.7,
        ramp=False,
    ),
}

def compute_fuzzy_losses(
    model: FuzzyBoxEmbedding,
    edges: OntologyEdges,
    cfg: FuzzyBoxConfig,
    step: int,
    schedule: Optional[CurriculumSchedule] = None,
) -> Tuple[torch.Tensor, dict]:
    """Compute all fuzzy losses with optional curriculum scheduling."""
    steps = cfg.steps
    def scheduled_weight(base_weight: float, start_frac: float) -> float:
        if schedule is None:
            return base_weight
        start_step = int(start_frac * steps)
        if step < start_step:
            return 0.0
        if not schedule.ramp or start_frac == 0.0:
            return base_weight
        progress = (step - start_step) / max(1, steps - start_step)
        return base_weight * min(max(progress, 0.0), 1.0)

    loss_dict = {}
    size_loss = cfg.size_weight * edges.min_box_loss(model, cfg)
    loss_dict['size'] = size_loss.item()
    total_loss = size_loss

    subclass_w = scheduled_weight(cfg.subclass_weight, schedule.subclass_start if schedule else 0.0)
    if subclass_w > 0 and edges.closure_sub_edges.numel() > 0:
        sub_loss = subclass_w * edges.subclass_loss(model)
        loss_dict['subclass'] = sub_loss.item()
        total_loss = total_loss + sub_loss
    else:
        loss_dict['subclass'] = 0.0

    sibling_w = scheduled_weight(cfg.distance_weight, schedule.sibling_start if schedule else 0.0)
    if sibling_w > 0:
        sib_loss = sibling_w * edges.distance_loss(model)
        loss_dict['sibling'] = sib_loss.item()
        total_loss = total_loss + sib_loss
    else:
        loss_dict['sibling'] = 0.0

    disjoint_w = scheduled_weight(cfg.disjoint_weight, schedule.disjoint_start if schedule else 0.0)
    if disjoint_w > 0:
        dis_loss = disjoint_w * edges.disjoint_loss(model, cfg=cfg)
        loss_dict['disjoint'] = dis_loss.item()
        total_loss = total_loss + dis_loss
    else:
        loss_dict['disjoint'] = 0.0

    bigbox_w = scheduled_weight(cfg.big_box_weight, schedule.big_box_start if schedule else 0.0)
    if bigbox_w > 0:
        big_loss = bigbox_w * edges.oversize_loss(model, cfg)
        loss_dict['big_box'] = big_loss.item()
        total_loss = total_loss + big_loss
    else:
        loss_dict['big_box'] = 0.0

    loss_dict['total'] = total_loss.item()
    return total_loss, loss_dict

def evaluate_fuzzy_satisfaction(
    model: FuzzyBoxEmbedding,
    edges: OntologyEdges,
    cfg: FuzzyBoxConfig,
    disjoint_margin: float = 0.02,
) -> dict:
    """Evaluate fuzzy embeddings using satisfaction degrees.
    
    Reports both:
    - Geometric violations (comparable to penalty-based models): actual box containment failures
    - Fuzzy violations (fuzzy-specific): edges below satisfaction threshold α
    """
    with torch.no_grad():
        result = {}
        
        # Get box boundaries for geometric violation counting
        mn, mx = model.get_min_max()
        
        # Subclass satisfaction
        if edges.closure_sub_edges.numel() > 0:
            child_idx = edges.closure_sub_edges[:, 0]
            parent_idx = edges.closure_sub_edges[:, 1]
            contain_degrees = model.containment_degree(child_idx, parent_idx)
            
            n_closure = len(contain_degrees)
            n_asserted = len(edges.asserted_sub_edges)
            
            # Fuzzy satisfaction metrics
            result['subclass_mean_satisfaction'] = contain_degrees.mean().item()
            result['subclass_min_satisfaction'] = contain_degrees.min().item()
            
            # Fuzzy violations at different thresholds
            for alpha in [0.9, 0.95, 0.99]:
                violations = (contain_degrees < alpha).sum().item()
                result[f'subclass_violations_alpha{alpha}'] = violations
                result[f'subclass_violations_alpha{alpha}_pct'] = violations / n_closure * 100
                
                # Average violation degree for edges below α
                violated_mask = contain_degrees < alpha
                if violated_mask.any():
                    avg_violation_degree = (1.0 - contain_degrees[violated_mask]).mean().item()
                    result[f'subclass_avg_violation_degree_alpha{alpha}'] = avg_violation_degree
                else:
                    result[f'subclass_avg_violation_degree_alpha{alpha}'] = 0.0
            
            # GEOMETRIC violations (comparable to penalty-based models)
            # Count EDGES where box containment actually fails (any dimension)
            child, parent = child_idx, parent_idx
            lower_violations = mn[child] < mn[parent]  # [n_edges, dim]
            upper_violations = mx[child] > mx[parent]  # [n_edges, dim]
            # An edge is violated if ANY dimension fails
            edge_violations = (lower_violations | upper_violations).any(dim=1)  # [n_edges]
            geometric_sub_violations = edge_violations.sum().item()
            
            result['subclass_violations_geometric'] = int(geometric_sub_violations)
            result['subclass_violations_geometric_pct'] = geometric_sub_violations / n_closure * 100
            
            # Asserted vs entailed geometric violations
            if len(edges.asserted_sub_edges) > 0:
                asserted_sub_tensor = torch.tensor(edges.asserted_sub_edges, dtype=torch.long, device=model.center.device)
                a_child = asserted_sub_tensor[:, 0]
                a_parent = asserted_sub_tensor[:, 1]
                
                # Fuzzy violations on asserted edges
                asserted_contain = model.containment_degree(a_child, a_parent)
                asserted_fuzzy_violations_95 = (asserted_contain < 0.95).sum().item()
                result['subclass_violations_asserted_fuzzy_alpha0.95'] = asserted_fuzzy_violations_95
                result['subclass_violations_asserted_fuzzy_alpha0.95_pct'] = asserted_fuzzy_violations_95 / n_asserted * 100
                
                # Geometric violations on asserted edges
                a_lower = (mn[a_child] < mn[a_parent]).any(dim=1).sum().item()
                a_upper = (mx[a_child] > mx[a_parent]).any(dim=1).sum().item()
                a_edge_violations = ((mn[a_child] < mn[a_parent]) | (mx[a_child] > mx[a_parent])).any(dim=1).sum().item()
                result['subclass_violations_asserted_geometric'] = int(a_edge_violations)
                result['subclass_violations_asserted_geometric_pct'] = a_edge_violations / n_asserted * 100
        else:
            result['subclass_mean_satisfaction'] = 1.0
            result['subclass_min_satisfaction'] = 1.0
            for alpha in [0.9, 0.95, 0.99]:
                result[f'subclass_violations_alpha{alpha}'] = 0
                result[f'subclass_violations_alpha{alpha}_pct'] = 0.0
                result[f'subclass_avg_violation_degree_alpha{alpha}'] = 0.0
            result['subclass_violations_asserted_fuzzy_alpha0.95'] = 0
            result['subclass_violations_asserted_fuzzy_alpha0.95_pct'] = 0.0
            result['subclass_violations_asserted_geometric'] = 0
            result['subclass_violations_asserted_geometric_pct'] = 0.0
            result['subclass_violations_geometric'] = 0
            result['subclass_violations_geometric_pct'] = 0.0
        
        # Disjoint satisfaction
        if edges.entailed_disjoint_edges.numel() > 0:
            idx_a = edges.entailed_disjoint_edges[:, 0]
            idx_b = edges.entailed_disjoint_edges[:, 1]
            overlap_degrees = model.overlap_degree(idx_a, idx_b)
            disjoint_satisfaction = 1.0 - overlap_degrees
            
            n_disj = len(disjoint_satisfaction)
            result['disjoint_mean_satisfaction'] = disjoint_satisfaction.mean().item()
            result['disjoint_min_satisfaction'] = disjoint_satisfaction.min().item()
            
            # Fuzzy violations at different thresholds
            for alpha in [0.9, 0.95, 0.99]:
                violations = (disjoint_satisfaction < alpha).sum().item()
                result[f'disjoint_violations_alpha{alpha}'] = violations
                result[f'disjoint_violations_alpha{alpha}_pct'] = violations / n_disj * 100
                
                # Average violation degree for edges below α
                violated_mask = disjoint_satisfaction < alpha
                if violated_mask.any():
                    avg_violation_degree = (1.0 - disjoint_satisfaction[violated_mask]).mean().item()
                    result[f'disjoint_avg_violation_degree_alpha{alpha}'] = avg_violation_degree
                else:
                    result[f'disjoint_avg_violation_degree_alpha{alpha}'] = 0.0
            
            # GEOMETRIC violations (comparable to penalty-based models)
            # Count pairs that don't have sufficient separation margin
            a, b = idx_a, idx_b
            sep_ab = mn[b] - mx[a]
            sep_ba = mn[a] - mx[b]
            sep = torch.maximum(sep_ab, sep_ba)
            # A pair is violated if max separation across all dimensions < margin
            geometric_dis_violations = (sep.max(dim=1).values < disjoint_margin).sum().item()
            
            result['disjoint_violations_geometric'] = int(geometric_dis_violations)
            result['disjoint_violations_geometric_pct'] = geometric_dis_violations / n_disj * 100
        else:
            result['disjoint_mean_satisfaction'] = 1.0
            result['disjoint_min_satisfaction'] = 1.0
            for alpha in [0.9, 0.95, 0.99]:
                result[f'disjoint_violations_alpha{alpha}'] = 0
                result[f'disjoint_violations_alpha{alpha}_pct'] = 0.0
                result[f'disjoint_avg_violation_degree_alpha{alpha}'] = 0.0
            result['disjoint_violations_geometric'] = 0
            result['disjoint_violations_geometric_pct'] = 0.0
        
        # Cardinality and temperature
        log_card = model.fuzzy_cardinality()
        result['avg_log_cardinality'] = log_card.mean().item()
        if model.raw_temperature is not None:
            temps = model.temperature()
            result['avg_temperature'] = temps.mean().item()
            result['max_temperature'] = temps.max().item()
        
        return result

def _build_fuzzy_df(
    model: FuzzyBoxEmbedding, edges: OntologyEdges, cfg: FuzzyBoxConfig
) -> pd.DataFrame:
    with torch.no_grad():
        center_np = model.center.cpu().numpy()
        half_size_np = model.half_size().cpu().numpy()
        temp_np = model.temperature().cpu().numpy() if model.raw_temperature is not None else None
    classes = [edges.id2cls[i] for i in range(edges.num_classes)]
    df_dict = {
        "class_uri": classes,
        "class_name": [_local_name(u) for u in classes],
    }
    for d in range(cfg.dim):
        df_dict[f"center_{d}"] = center_np[:, d]
        df_dict[f"half_size_{d}"] = half_size_np[:, d]
        if temp_np is not None:
            df_dict[f"temp_{d}"] = temp_np[:, d]
    return pd.DataFrame(df_dict).sort_values("class_name").reset_index(drop=True)

def learn_fuzzy_boxes(
    owl_path: str,
    cfg: FuzzyBoxConfig,
    schedule: Optional[CurriculumSchedule] = None,
    device: Optional[str] = None,
    _preloaded=None,
    steps: Optional[int] = None,
    save_training_curve: bool = True,
) -> Tuple[FuzzyBoxEmbedding, pd.DataFrame, OntologyEdges, dict]:
    """Learn fuzzy box embeddings with optional curriculum scheduling.
    
    Returns:
        Tuple of (model, dataframe, edges, info_dict) where info_dict contains:
            - loss: final loss value
            - training_curve: list of (step, loss) tuples if save_training_curve=True
            - train_time: total training time in seconds
    """
    schedule = None
    
    device = device or "mps"
    if steps is None:
        steps = cfg.steps
    # Ensure lr is computed
    if cfg.lr is None:
        cfg.lr = 0.1 / math.sqrt(cfg.dim)
    
    classes, subclass_of, disjoint_pairs = _preloaded or load_owl(owl_path)
    edges = OntologyEdges(classes, subclass_of, disjoint_pairs, device=device)
    model = FuzzyBoxEmbedding(len(classes), cfg.dim, cfg, seed=cfg.seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    
    # Compute OWL file checksum
    try:
        with open(owl_path, "rb") as f:
            owl_sha256 = hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        owl_sha256 = None
    
    training_curve = [] if save_training_curve else None
    train_start = time.time()
    final_loss = None
    
    for step in range(1, steps + 1):
        opt.zero_grad()
        total_loss, loss_dict = compute_fuzzy_losses(model, edges, cfg, step, schedule)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        final_loss = total_loss.item()
        
        if save_training_curve and step % max(1, steps // 20) == 0:
            training_curve.append({"step": step, "loss": final_loss})
        
        if step % max(1, steps // 10) == 0:
            metrics = evaluate_fuzzy_satisfaction(model, edges, cfg)
            log_msg = (
                f"step {step:>5}/{steps} | loss={final_loss:.4f} | "
                f"sub_sat={metrics['subclass_mean_satisfaction']:.3f} | "
                f"dis_sat={metrics['disjoint_mean_satisfaction']:.3f}"
            )
            if schedule:
                log_msg += (
                    f" | sub_loss={loss_dict.get('subclass', 0):.3f} | "
                    f"disj_loss={loss_dict.get('disjoint', 0):.3f}"
                )
            print(log_msg)
    
    train_time = time.time() - train_start
    
    with torch.no_grad():
        metrics = evaluate_fuzzy_satisfaction(model, edges, cfg)
        edges.last_subclass_violations = metrics.get('subclass_violations_alpha0.95', 0)
        edges.last_disjoint_violations = metrics.get('disjoint_violations_alpha0.95', 0)
    
    # Get git commit hash
    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(owl_path) or "."
        ).decode().strip()
    except Exception:
        git_rev = None
    
    return model, _build_fuzzy_df(model, edges, cfg), edges, {
        "loss": final_loss,
        "training_curve": training_curve,
        "train_time": train_time,
        "git_rev": git_rev,
        "owl_sha256": owl_sha256,
    }

def _train_one_dim(args: tuple) -> tuple[int, dict]:
    """Train one dimension for sweep_dimensions()."""
    d, owl_path, device, learn_fn, classes, subclass_of, disjoint_pairs, schedule, cfg, steps = args
    
    if cfg is None:
        cfg = FuzzyBoxConfig(dim=d, steps=3000)
    else:
        cfg = replace(cfg, dim=d)
    
    kwargs = dict(
        owl_path=owl_path,
        cfg=cfg,
        device=device,
        _preloaded=(classes, subclass_of, disjoint_pairs),
        steps=steps,
    )
    
    if schedule is not None:
        kwargs["schedule"] = schedule
    
    train_start = time.time()
    result = learn_fn(**kwargs)
    train_time = time.time() - train_start
    
    if len(result) == 4:
        model, df, edges, info = result
        if isinstance(info, dict):
            loss = info.get("loss")
            training_curve = info.get("training_curve")
            git_rev = info.get("git_rev")
            owl_sha256 = info.get("owl_sha256")
            train_time_actual = info.get("train_time", train_time)
        else:
            loss = info
            training_curve = None
            git_rev = None
            owl_sha256 = None
            train_time_actual = train_time
    else:
        model, df, edges = result
        loss = None
        training_curve = None
        git_rev = None
        owl_sha256 = None
        train_time_actual = train_time
    
    # Get full metrics
    metrics = evaluate_fuzzy_satisfaction(model, edges, cfg)
    
    return d, {
        "model": model,
        "df": df,
        "edges": edges,
        "cfg": cfg,
        "loss": loss,
        "train_time": train_time_actual,
        "training_curve": training_curve,
        "git_rev": git_rev,
        "owl_sha256": owl_sha256,
        "sub_viol": metrics.get('subclass_violations_alpha0.95', 0),
        "dis_viol": metrics.get('disjoint_violations_alpha0.95', 0),
        "subclass_violations_geometric": metrics.get('subclass_violations_geometric', 0),
        "disjoint_violations_geometric": metrics.get('disjoint_violations_geometric', 0),
        "subclass_violations_geometric_pct": metrics.get('subclass_violations_geometric_pct', 0),
        "disjoint_violations_geometric_pct": metrics.get('disjoint_violations_geometric_pct', 0),
        "subclass_violations_alpha0.95_pct": metrics.get('subclass_violations_alpha0.95_pct', 0),
        "disjoint_violations_alpha0.95_pct": metrics.get('disjoint_violations_alpha0.95_pct', 0),
        "subclass_mean_satisfaction": metrics.get('subclass_mean_satisfaction', 0),
        "disjoint_mean_satisfaction": metrics.get('disjoint_mean_satisfaction', 0),
        "subclass_avg_violation_degree_alpha0.95": metrics.get('subclass_avg_violation_degree_alpha0.95', 0),
        "disjoint_avg_violation_degree_alpha0.95": metrics.get('disjoint_avg_violation_degree_alpha0.95', 0),
    }


def sweep_dimensions(
        owl_path: str,
        learn_fn: Callable = learn_fuzzy_boxes,
        dims: Iterable[int] = range(2, 11),
        device: Optional[str] = None,
        max_workers: Optional[int] = None,
        schedule=None,
        cfg=None,
        path: Optional[str] = None,
        steps: Optional[int] = None,
) -> Dict[int, Dict]:
    """Sweep across multiple embedding dimensions.
    
    Reuses the structure of the legacy penalty-era sweep_dimensions.
    """
    device = device or "mps"
    dims = list(dims)
    
    print(f"Loading OWL: {owl_path}")
    
    # ------------------ timing ------------------
    load_start = time.time()
    classes, subclass_of, disjoint_pairs = load_owl(owl_path)
    load_time = time.time() - load_start
    print(
        f"OWL loaded in {timedelta(seconds=int(load_time))} ({load_time:.2f}s)"
    )
    # --------------------------------------------
    
    print(
        f"Sweeping dims={dims} | device={device} | schedule={'plain' if schedule is None else 'curriculum'}"
    )
    
    if path is not None:
        os.makedirs(path, exist_ok=True)
    
    results: Dict[int, Dict] = {}
    dim_times = []
    
    def _save_single(dim: int, info: dict):
        """Save one dimension in both old (pkl) and new (json) formats."""
        if path is None:
            return
        
        dim_path = os.path.join(path, f"dim_{dim}")
        os.makedirs(dim_path, exist_ok=True)
        
        raw_model = getattr(info["model"], "_orig_mod", info["model"])
        state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in raw_model.state_dict().items()
        }
        
        tmp_path = os.path.join(dim_path, "model.pt.tmp")
        final_path = os.path.join(dim_path, "model.pt")
        
        torch.save(state_dict, tmp_path)
        os.replace(tmp_path, final_path)
        
        with open(os.path.join(dim_path, "data.pkl"), "wb") as f:
            pickle.dump(
                {
                    "df": info["df"],
                    "sub_viol": info["sub_viol"],
                    "dis_viol": info["dis_viol"],
                    "loss": info.get("loss"),
                },
                f,
            )
        
        df = info["df"]
        num_classes = len(df)
        
        model = info["model"]
        edges = info.get("edges")
        
        with torch.no_grad():
            mn, mx = model.get_min_max()
            mn = mn.clamp(-1e6, 1e6)
            mx = mx.clamp(-1e6, 1e6)
            
            box_sizes = (mx - mn).mean(dim=1)
            volumes = model.volumes()
        
        closure_check = {}
        if edges is not None:
            closure_check = edges.check_entailed_and_closure_edges(mn, mx, 0.02)
        
        # Environment and runtime metadata
        import sys
        
        # Get current timestamp
        timestamp = datetime.now().isoformat()
        
        # Box volume distribution percentiles
        box_sizes_np = box_sizes.cpu().numpy()
        volumes_np = volumes.cpu().numpy()
        box_percentiles = {
            "p10": float(np.percentile(box_sizes_np, 10)),
            "p50": float(np.percentile(box_sizes_np, 50)),
            "p90": float(np.percentile(box_sizes_np, 90)),
        }
        volume_percentiles = {
            "p10": float(np.percentile(volumes_np, 10)),
            "p50": float(np.percentile(volumes_np, 50)),
            "p90": float(np.percentile(volumes_np, 90)),
        }
        
        metrics = {
            "run_name": os.path.basename(os.path.normpath(path)),
            "ontology": os.path.basename(os.path.dirname(os.path.normpath(path))),
            "owl_path": owl_path,  # Full path for reproducibility
            "owl_file_sha256": info.get("owl_sha256"),
            "dimension": dim,
            "variant": "plain" if schedule is None else "curriculum",
            "steps": steps if steps is not None else (cfg.steps if cfg else 3000),
            "curriculum": schedule is not None,
            "device": device,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "timestamp": timestamp,
            "git_rev": info.get("git_rev"),  # Git commit hash
            "num_classes": num_classes,
            "num_subclass_axioms": len(subclass_of),
            "num_disjoint_pairs": len(disjoint_pairs),
            "final_loss": float(info.get("loss", 0))
            if info.get("loss")
            else None,
            "train_time_seconds": float(info.get("train_time", 0)),
            "subclass_violations": int(info["sub_viol"]),
            "disjoint_violations": int(info["dis_viol"]),
            "subclass_violations_geometric": int(info.get("subclass_violations_geometric", 0)),
            "disjoint_violations_geometric": int(info.get("disjoint_violations_geometric", 0)),
            "subclass_mean_satisfaction": float(info.get("subclass_mean_satisfaction", 0)),
            "disjoint_mean_satisfaction": float(info.get("disjoint_mean_satisfaction", 0)),
            "subclass_avg_violation_degree_alpha0.95": float(info.get("subclass_avg_violation_degree_alpha0.95", 0)),
            "disjoint_avg_violation_degree_alpha0.95": float(info.get("disjoint_avg_violation_degree_alpha0.95", 0)),
            "avg_box_size": float(box_sizes.mean().item()),
            "box_size_percentiles": box_percentiles,
            "avg_log_volume": float(volumes.mean().item()),
            "log_volume_percentiles": volume_percentiles,
            "min_box_size": float(box_sizes.min().item()),
            "max_box_size": float(box_sizes.max().item()),
            "closure_subclass_total": closure_check.get("closure_sub_total"),
            "closure_subclass_found": closure_check.get("closure_sub_found"),
            "closure_subclass_rate": float(
                closure_check.get("closure_sub_rate", 0)
            )
            if closure_check.get("closure_sub_rate")
            else None,
            "entailed_disjoint_total": closure_check.get(
                "entailed_dis_total"
            ),
            "entailed_disjoint_found": closure_check.get(
                "entailed_dis_found"
            ),
            "entailed_disjoint_rate": float(
                closure_check.get("entailed_dis_rate", 0)
            )
            if closure_check.get("entailed_dis_rate")
            else None,
        }


        ch7_files = {}
        if edges is not None:
            import csv as _csv
            rows, cal = [], []
            with torch.no_grad():
                if edges.closure_sub_edges is not None and edges.closure_sub_edges.numel() > 0:
                    deg = model.containment_degree(
                        edges.closure_sub_edges[:, 0], edges.closure_sub_edges[:, 1]
                    ).cpu().numpy()
                    for i, v in enumerate(deg):
                        rows.append((f"sub_{i}", "subclass", float(v)))
                        cal.append((f"sub_{i}", float(v), int(v > 0.5)))
                if edges.entailed_disjoint_edges is not None and edges.entailed_disjoint_edges.numel() > 0:
                    deg = model.overlap_degree(
                        edges.entailed_disjoint_edges[:, 0], edges.entailed_disjoint_edges[:, 1]
                    ).cpu().numpy()
                    for i, v in enumerate(deg):
                        d = 1.0 - float(v)
                        rows.append((f"dis_{i}", "disjoint", d))
                        cal.append((f"dis_{i}", d, int(d > 0.5)))
            if rows:
                _sat = os.path.join(dim_path, "satisfaction_degrees.csv")
                with open(_sat, "w", newline="") as f:
                    w = _csv.writer(f)
                    w.writerow(["axiom_id", "axiom_type", "satisfaction_degree"])
                    w.writerows(rows)
                _cal = os.path.join(dim_path, "calibration_pairs.csv")
                with open(_cal, "w", newline="") as f:
                    w = _csv.writer(f)
                    w.writerow(["axiom_id", "predicted_confidence", "actual_satisfied"])
                    w.writerows(cal)
                ch7_files = {
                    "satisfaction_degrees_csv": "satisfaction_degrees.csv",
                    "calibration_pairs_csv": "calibration_pairs.csv",
                }
        metrics["files_saved"] = ch7_files
        metrics["chapter_7_ready"] = bool(ch7_files)

        with open(os.path.join(dim_path, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save training curve if available (for paper plots)
        if info.get("training_curve"):
            with open(os.path.join(dim_path, "training_curve.json"), "w") as f:
                json.dump(info["training_curve"], f, indent=2)
        
        # Save config for reproducibility
        save_cfg = info.get("cfg") or cfg  # prefer per-dim cfg from _train_one_dim
        assert save_cfg is not None, "cfg must be provided"
        config_info = {
            "dim": save_cfg.dim,
            "steps": steps if steps is not None else save_cfg.steps,
            "lr": save_cfg.lr,
            "seed": save_cfg.seed,
            "membership_type": save_cfg.membership_type,
            "t_norm": save_cfg.t_norm,
            "implication": save_cfg.implication,
            "learn_temperature": save_cfg.learn_temperature,
            "init_temperature": save_cfg.init_temperature,
            "min_box_size": save_cfg.min_box_size,
            "size_weight": save_cfg.size_weight,
            "subclass_weight": save_cfg.subclass_weight,
            "disjoint_weight": save_cfg.disjoint_weight,
            "contact_weight": save_cfg.contact_weight,
            "contact_on_entailed": save_cfg.contact_on_entailed,
            "big_box_weight": save_cfg.big_box_weight,
            "distance_weight": save_cfg.distance_weight,
            "base_log_volume": save_cfg.base_log_volume,
            "depth_scale": save_cfg.depth_scale,
            "min_log_volume": save_cfg.min_log_volume,
        }
        with open(os.path.join(dim_path, "config.json"), "w") as f:
            json.dump(config_info, f, indent=2)
        
        # Save schedule info (always save, even for PLAIN)
        if schedule is not None:
            schedule_info = {
                "disjoint_start": schedule.disjoint_start,
                "subclass_start": schedule.subclass_start,
                "sibling_start": schedule.sibling_start,
                "big_box_start": schedule.big_box_start,
                "ramp": schedule.ramp,
            }
        else:
            schedule_info = {
                "disjoint_start": 0.0,
                "subclass_start": 0.0,
                "sibling_start": 0.0,
                "big_box_start": 0.0,
                "ramp": False,
                "name": "PLAIN",
            }
        with open(os.path.join(dim_path, "schedule.json"), "w") as f:
            json.dump(schedule_info, f, indent=2)
    
    for d in dims:
        print("=" * 60)
        print(f"Training dim={d} with {learn_fn.__name__}")
        
        start_dim = time.time()
        
        _, result = _train_one_dim(
            (
                d,
                owl_path,
                device,
                learn_fn,
                classes,
                subclass_of,
                disjoint_pairs,
                schedule,
                cfg,
                steps,
            )
        )
        
        elapsed_dim = time.time() - start_dim
        dim_times.append(elapsed_dim)
        
        results[d] = result
        _save_single(d, result)
        
        print(f"\nResults ({elapsed_dim:.1f}s):")
        if schedule is not None:
            print(f"  Schedule: disjoint_start={schedule.disjoint_start*100:.1f}%, subclass_start={schedule.subclass_start*100:.1f}%, sibling_start={schedule.sibling_start*100:.1f}%")
        print(f"  Final loss: {result['loss']:.4f}")
        print(f"  Subclass sat: {result['subclass_mean_satisfaction']:.3f}")
        print(f"  Disjoint sat: {result['disjoint_mean_satisfaction']:.3f}")
        print(f"  Geometric violations (penalty-comparable): sub={result['subclass_violations_geometric']} ({result['subclass_violations_geometric_pct']:.1f}%), disj={result['disjoint_violations_geometric']} ({result['disjoint_violations_geometric_pct']:.1f}%)")
        print(f"  Fuzzy violations (α=0.95): sub={result['sub_viol']} ({result.get('subclass_violations_alpha0.95_pct', 0):.1f}%), disj={result['dis_viol']} ({result.get('disjoint_violations_alpha0.95_pct', 0):.1f}%)")
        print(f"  Avg violation degree (α=0.95): sub={result['subclass_avg_violation_degree_alpha0.95']:.4f}, disj={result['disjoint_avg_violation_degree_alpha0.95']:.4f}")

    avg_dim_time = (
        sum(dim_times) / len(dim_times) if dim_times else 0.0
    )
    
    print("\n" + "=" * 60)
    print(
        f"OWL loading time : "
        f"{timedelta(seconds=int(load_time))} ({load_time:.2f}s)"
    )
    print(
        f"Average per dim  : "
        f"{timedelta(seconds=int(avg_dim_time))} ({avg_dim_time:.2f}s)"
    )
    print(f"Finished sweep. Stored {len(results)} dimensions.")
    print("=" * 60)
    
    return dict(sorted(results.items()))


def fast_load_sweep_results(path, load_per_axiom=False):
    """Fast-load sweep results by extracting only metrics.json files.
    
    Skips model loading entirely — returns just the metrics dictionaries.
    
    Args:
        path: Base directory containing dim_* subdirectories with metrics.json
        load_per_axiom: If True, also load per_axiom_violations.json when available
    
    Returns:
        Dict mapping dimension (int) -> dict with keys:
            - metrics: dict from metrics.json
            - config: dict from config.json (or None if missing)
            - schedule: dict from schedule.json (or None if missing)
            - training_curve: list from training_curve.json (or None if missing)
            - per_axiom: list (only if load_per_axiom=True and file exists)
    """
    results = {}
    dim_dirs = sorted([d for d in os.listdir(path) if d.startswith("dim_")])
    
    for d in dim_dirs:
        dim_path = os.path.join(path, d)
        metrics_path = os.path.join(dim_path, "metrics.json")
        config_path = os.path.join(dim_path, "config.json")
        schedule_path = os.path.join(dim_path, "schedule.json")
        training_curve_path = os.path.join(dim_path, "training_curve.json")
        per_axiom_path = os.path.join(dim_path, "per_axiom_violations.json")
        
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            
            # Extract dimension from dirname or metrics
            dim = int(d.replace("dim_", ""))
            
            result_entry = {"metrics": metrics}
            
            # Load config.json (new)
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    result_entry["config"] = json.load(f)
            else:
                result_entry["config"] = None
            
            # Load schedule.json
            if os.path.exists(schedule_path):
                with open(schedule_path, "r") as f:
                    result_entry["schedule"] = json.load(f)
            else:
                result_entry["schedule"] = None
            
            # Load training_curve.json
            if os.path.exists(training_curve_path):
                with open(training_curve_path, "r") as f:
                    result_entry["training_curve"] = json.load(f)
            else:
                result_entry["training_curve"] = None
            
            if load_per_axiom and os.path.exists(per_axiom_path):
                with open(per_axiom_path, "r") as f:
                    per_axiom_data = json.load(f)
                result_entry["per_axiom"] = per_axiom_data.get("per_axiom")
                print(f"  ✓ Loaded metrics + config + schedule + training_curve + per_axiom (dim={dim})")
            else:
                print(f"  ✓ Loaded metrics + config + schedule + training_curve (dim={dim})")
            
            results[dim] = result_entry
        else:
            print(f" No metrics.json in {d}")
    
    print(f"Loaded metrics for {len(results)} dimensions from {path}")
    return results


def load_sweep_results(path, load_models=True):
    """Load full sweep results including models.
    
    Args:
        path: Base directory containing dim_* subdirectories
        load_models: If True, also load model.pt files
    
    Returns:
        Dict mapping dimension (int) -> dict with keys:
            - model: FuzzyBoxEmbedding (if load_models=True)
            - df: pandas DataFrame
            - metrics: dict from metrics.json
            - config: dict from config.json (or None)
            - schedule: dict from schedule.json (or None)
            - training_curve: list from training_curve.json (or None)
            - data: dict from data.pkl
    """
    import glob
    
    results = {}
    dim_dirs = sorted([d for d in os.listdir(path) if d.startswith("dim_")])
    
    for d in dim_dirs:
        dim_path = os.path.join(path, d)
        dim = int(d.replace("dim_", ""))
        
        result = {}
        
        # Load metrics
        metrics_path = os.path.join(dim_path, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                result["metrics"] = json.load(f)
        
        # Load config.json
        config_path = os.path.join(dim_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                result["config"] = json.load(f)
        else:
            result["config"] = None
        
        # Load schedule.json
        schedule_path = os.path.join(dim_path, "schedule.json")
        if os.path.exists(schedule_path):
            with open(schedule_path, "r") as f:
                result["schedule"] = json.load(f)
        else:
            result["schedule"] = None
        
        # Load training_curve.json
        training_curve_path = os.path.join(dim_path, "training_curve.json")
        if os.path.exists(training_curve_path):
            with open(training_curve_path, "r") as f:
                result["training_curve"] = json.load(f)
        else:
            result["training_curve"] = None
        
        # Load data.pkl
        data_path = os.path.join(dim_path, "data.pkl")
        if os.path.exists(data_path):
            with open(data_path, "rb") as f:
                result["data"] = pickle.load(f)
        
        # Load model
        if load_models:
            model_path = os.path.join(dim_path, "model.pt")
            if os.path.exists(model_path):
                # Load on the device the model was trained on (recorded in
                # metrics.json), so downstream evaluation against edges on the
                # same device works without manual .to() calls.
                load_device = result.get("metrics", {}).get("device", "cpu")
                state_dict = torch.load(model_path, map_location=load_device)
                
                # Infer dim from state dict
                center_shape = state_dict.get("center", torch.tensor(0)).shape
                inferred_dim = center_shape[1] if len(center_shape) > 1 else 2
                
                # Need class count from metrics or config
                num_classes = result.get("metrics", {}).get("num_classes", 0)
                
                if num_classes > 0:
                    # Reconstruct config from config.json if available
                    if result["config"] is not None:
                        cfg_dict = result["config"]
                        cfg = FuzzyBoxConfig(
                            dim=cfg_dict.get("dim", inferred_dim),
                            steps=cfg_dict.get("steps", 5000),
                            lr=cfg_dict.get("lr", 0.05),
                            seed=cfg_dict.get("seed", 42),
                            membership_type=cfg_dict.get("membership_type", "sigmoidal"),
                            t_norm=cfg_dict.get("t_norm", "product"),
                            implication=cfg_dict.get("implication", "reichenbach"),
                            learn_temperature=cfg_dict.get("learn_temperature", False),
                            init_temperature=cfg_dict.get("init_temperature", 0.1),
                            min_box_size=cfg_dict.get("min_box_size", 0.05),
                            size_weight=cfg_dict.get("size_weight", 0.1),
                            subclass_weight=cfg_dict.get("subclass_weight", 2.0),
                            disjoint_weight=cfg_dict.get("disjoint_weight", 2.0),
                            contact_weight=cfg_dict.get("contact_weight", 0.5),
                            contact_on_entailed=cfg_dict.get("contact_on_entailed", False),
                            big_box_weight=cfg_dict.get("big_box_weight", 0.1),
                            distance_weight=cfg_dict.get("distance_weight", 0.0),
                            base_log_volume=cfg_dict.get("base_log_volume", 2.0),
                            depth_scale=cfg_dict.get("depth_scale", 0.5),
                            min_log_volume=cfg_dict.get("min_log_volume", -4.0),
                        )
                    else:
                        cfg = FuzzyBoxConfig(dim=inferred_dim)
                    
                    model = FuzzyBoxEmbedding(num_classes, inferred_dim, cfg)
                    model.load_state_dict(state_dict)
                    model = model.to(load_device)  # match trained device (mps)
                    result["model"] = model
                    print(f"  ✓ Loaded model + metrics + config + schedule + training_curve (dim={dim})")
                else:
                    print(f"  ⚠ Cannot load model: unknown num_classes (dim={dim})")
            else:
                print(f"  ⚠ No model.pt in {d}")
        else:
            print(f"  ✓ Loaded metrics + config + schedule + training_curve (dim={dim})")
        
        results[dim] = result
    
    print(f"Loaded {len(results)} dimensions from {path}")
    return results
