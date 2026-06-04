"""MJCF helpers for multi-ant maze: build N ants from template and collision filtering."""

from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ET

# MuJoCo uses integer bitmasks; reserve bit 0 for world (floor/walls), bit (i+1) for agent i.
_MAX_AGENTS_COLLISION = 28


def world_agent_conaffinity_mask(n_agents: int) -> int:
    """Bitmask for floor/wall ``conaffinity``: collide with every agent's foot ``contype`` bit."""
    if n_agents < 1:
        raise ValueError("n_agents must be >= 1")
    if n_agents > _MAX_AGENTS_COLLISION:
        raise ValueError(
            f"n_agents={n_agents} too large for collision bit assignment (max {_MAX_AGENTS_COLLISION})"
        )
    mask = 0
    for k in range(n_agents):
        mask |= 1 << (k + 1)
    return mask


def _init_qpos_numbers_for_n_agents(n_agents: int, dx: float = 2.0) -> str:
    """init_qpos: per ant 15 floats (free joint xy + rest), then target slide (2)."""
    row = [0.0, 0.0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
    parts = []
    for i in range(n_agents):
        r = list(row)
        r[0], r[1] = float(i * dx), 0.0
        parts.extend(r)
    parts.extend([0.0, 0.0])
    return " ".join(str(x) for x in parts)


def _apply_agent_only_collision_to_torso(torso: ET.Element, agent_index: int) -> None:
    """World bit 1; agent k uses contype 1<<(k+1). Agents do not collide with each other."""
    world_bit = 1
    agent_bit = 1 << (agent_index + 1)
    for geom in torso.iter("geom"):
        ct_raw = geom.get("contype")
        ca_raw = geom.get("conaffinity")
        contype = int(ct_raw) if ct_raw not in (None, "") else 0
        conaffinity = int(ca_raw) if ca_raw not in (None, "") else 0
        # Template feet: contype=1, conaffinity=1 (collide only with world in the new scheme).
        if contype == 1 and conaffinity == 1:
            geom.set("contype", str(agent_bit))
            geom.set("conaffinity", str(world_bit))
        # Torso / aux capsules: receive contacts from this agent's feet only (+ world).
        elif contype == 0 and conaffinity == 1:
            geom.set("conaffinity", str(world_bit | agent_bit))


def _configure_floor_collision(worldbody: ET.Element, n_agents: int) -> None:
    mask = world_agent_conaffinity_mask(n_agents)
    for geom in worldbody.findall("geom"):
        if geom.get("name") == "floor":
            geom.set("contype", "1")
            geom.set("conaffinity", str(mask))
            return
    raise RuntimeError("floor geom not found under worldbody")


def default_multi_ant_template_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "multi_ant_maze.xml")


def build_multi_ant_maze_root(
    n_agents: int,
    template_path: str | None = None,
) -> ET.Element:
    """Load the ant template and rebuild MJCF with ``n_agents`` copies (names + actuators + init_qpos).

    Agent--agent contacts are disabled via ``contype`` / ``conaffinity`` bitmasks; each ant still
    collides with the floor, maze blocks, and its own body (feet vs torso/aux).
    """
    if n_agents < 1:
        raise ValueError("n_agents must be >= 1")
    if template_path is None:
        template_path = default_multi_ant_template_path()
    tree = ET.parse(template_path)
    root = tree.getroot()
    worldbody = root.find(".//worldbody")
    actuator = root.find("actuator")
    if worldbody is None or actuator is None:
        raise RuntimeError(f"Invalid template: {template_path}")

    motors_a = [
        copy.deepcopy(m)
        for m in actuator
        if m.tag == "motor" and m.get("joint", "").endswith("_a")
    ]
    if len(motors_a) != 8:
        raise RuntimeError(f"Expected 8 motors for ant `_a` in template, got {len(motors_a)}")

    torso_a = None
    for child in list(worldbody):
        if child.tag == "body" and child.get("name") == "torso_a":
            torso_a = child
            break
    if torso_a is None:
        raise RuntimeError("torso_a not found in multi_ant_maze.xml template")

    for child in list(worldbody):
        if child.tag == "body" and child.get("name", "").startswith("torso_"):
            worldbody.remove(child)

    for m in list(actuator):
        actuator.remove(m)

    target = worldbody.find("./body[@name='target']")
    if target is None:
        raise RuntimeError("target body not found")
    insert_at = list(worldbody).index(target)

    dx = 2.0
    for i in range(n_agents):
        body = copy.deepcopy(torso_a)
        for el in body.iter():
            for key in ("name", "joint"):
                val = el.get(key)
                if val is not None and "_a" in val:
                    el.set(key, val.replace("_a", f"_{i}"))
        body.set("pos", f"{i * dx:.1f} 0 0.75")
        _apply_agent_only_collision_to_torso(body, i)
        worldbody.insert(insert_at + i, body)

    for i in range(n_agents):
        for tmpl in motors_a:
            m = copy.deepcopy(tmpl)
            j = m.get("joint")
            if j is not None:
                m.set("joint", j.replace("_a", f"_{i}"))
            actuator.append(m)

    init_el = root.find(".//numeric[@name='init_qpos']")
    if init_el is not None:
        init_el.set("data", _init_qpos_numbers_for_n_agents(n_agents, dx=dx))

    _configure_floor_collision(worldbody, n_agents)

    return root
