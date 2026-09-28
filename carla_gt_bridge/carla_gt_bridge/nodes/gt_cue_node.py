"""Answer brain's cues from GROUND TRUTH instead of a camera.

    /carla/actor_list   (carla_msgs/CarlaActorList)  ->  is a cone in the junction?
    /predicted_cluster  (std_msgs/Int16)             ->  are we at an intersection?
                                                     ->  /cue/confirmations (String JSON)

Pairs with ``brain_controller``'s ``cue_source:=topic``. Everything downstream of the
answer — the ordinal de-bounce, the cue timeout, the branch pick — is unchanged, because
all of it consumes a single boolean per cue. That is the point of the seam: Phase A
answers from the simulator's own state, Phase C answers from the VLM, and the offline
replay answers from cached VLM responses, with identical machinery in between.

Why the cone is answered from the actor list and not the camera
---------------------------------------------------------------
The hard mission's branch is "turn right at the second intersection, unless there is a
cone in it". Phase A has no camera at all (``no_rendering_mode`` is on), but the cone is a
real actor in the world, so the *decision* can be exercised for real — spawn the prop and
the plan takes one path, destroy it and the plan takes the other. Running the same mission
twice, with the world changed by a ROS service, exercises both sides of the branch. Phase
C runs it with a camera and a VLM, and the only thing that changes is where the boolean
comes from.

Scoping matters: a cone anywhere in the town must not answer "yes". The cone counts only
when it is within ``cone_radius_m`` of the ego AND the ego is in a junction, which is the
best available stand-in for "the VLM can see it in the intersection ahead".
"""

from __future__ import annotations

import glob
import json

from carla_gt_bridge import cue_answers
import math
import os

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Int16, String

from ..region_lookup import load_region_table

#: Any actor whose type_id contains one of these is "a cone" for cue purposes.
CONE_TYPES = ("constructioncone", "trafficcone", "cone")

#: landmark family -> actor type_id substrings that count as one.
#:
#: Everything here is answered the same way the cone always was: from
#: ``/carla/actor_list``, which carries types, scoped by distance. Phase A runs with
#: ``no_rendering_mode`` on and has no camera, so ground truth stands in for perception
#: and the PLAN side stays testable before the perception side exists.
#:
#: A family absent from this table (or present with no actors in the town) is reported as
#: None, NOT False -- an unanswerable cue must time the step out rather than look like a
#: confident negative. See cue_answers.answer_keys.
#: Every substring here was checked against the REAL blueprint library
#: (test/fixtures/carla_static_props.json, CARLA 0.9.14, 95 props) by
#: test_every_family_substring_matches_a_real_blueprint and
#: test_no_blueprint_matches_two_families. Both checks matter:
#: a substring that matches NOTHING is a family that can never answer, and a blueprint
#: matching TWO families is resolved by `next()` over dict order -- i.e. silently, and
#: differently the moment someone reorders this table.
#:
#: `busstop` contains "stop" and is the reason stop_sign is keyed on "stopsign" /
#: "traffic.stop" and never on a bare "stop".
LANDMARK_ACTOR_TYPES: dict[str, tuple[str, ...]] = {
    "bench":         ("bench",),
    "bus_shelter":   ("busstop", "bus_stop", "busshelter"),
    # spawnable props -- see cue_answers.LANDMARK_SPELLINGS for why these
    # families and not others.
    "fountain":      ("fountain",),
    "kiosk":         ("kiosk", "foodcart"),
    "vending_machine": ("vendingmachine", "atm"),
    "trash_can":     ("trashcan",),
    "recycling_container": ("container",),
    "barrier":       ("barrier",),
    "construction_sign": ("warning",),
    "advertisement": ("advertisement",),
    "haybale":       ("haybale",),
    "mailbox":       ("mailbox",),
    "stop_sign":     ("traffic.stop", "stopsign", "stop_sign"),
    "traffic_light": ("traffic_light", "trafficlight"),
    # MAP STRUCTURES: not actors at all. Answered only from the region lists in
    # config/landmarks.<town>.json (see _load_native_landmarks); the empty tuple means the
    # actor list is never consulted for them.
    "overpass":      (),
    "car_park":      (),
}


class GtCueNode(Node):
    def __init__(self) -> None:
        super().__init__("gt_cue_node")
        self.declare_parameter("regions_npz", "")
        # Publish the place keys BEFORE the object keys, reproducing the ordering that
        # made every ambiguous branch cue answer "am I at a junction" instead of "is
        # there a cone". Experimental switch.
        self.declare_parameter("cue_place_first", False)
        self.declare_parameter("actor_list_topic", "/carla/actor_list")
        self.declare_parameter("odom_topic", "/carla/ego_vehicle/odometry")
        self.declare_parameter("cue_topic", "/cue/confirmations")
        self.declare_parameter("cone_radius_m", 40.0)
        self.declare_parameter("publish_hz", 4.0)
        # Force the cone answer without a cone actor existing.
        #   "auto"    (default) answer from the actor list — the real path
        #   "present" / "absent"  assert it
        #
        # The cone's whole job is to make one boolean true. Spawning the prop is the
        # stronger test because the answer travels the full path (actor list -> here ->
        # brain's branch pick), but it is not the test the BRANCH needs, and prop spawning
        # can race in the bridge. Forcing the boolean separates "does the branch mechanism
        # work" from "did the prop spawn".
        self.declare_parameter("force_cone", "auto")
        # Regions where cones were actually placed, e.g. [66, 53]. Empty = derive from the
        # actor list, which is the real path.
        #
        # This exists because static props publish no pose: CarlaActorInfo has no
        # transform and /carla/objects does not list them, so the actor list can say a
        # cone EXISTS but never where. Without positions the cue degrades to
        # "a cone exists somewhere AND I am at a junction", which for a corridor whose
        # junctions all have cones is indistinguishable from "I am at a junction" — the
        # ordinal would count correctly for the wrong reason.
        #
        # Phase A ground truth is entitled to know where we put things; the run script
        # sets this to the same regions it spawned into, so the two cannot disagree.
        # Comma-separated, e.g. "66,53". A STRING, not a list: rclpy cannot infer the
        # type of an empty array default, and "no cones configured" has to be
        # expressible without inventing a sentinel id.
        self.declare_parameter("cone_regions", "")
        #: cones/props tables, so a SPAWNED region can be attributed to a landmark family.
        #: Without this a bench answer needs a pose from /carla/objects, and that topic does
        #: not list static props in this bridge build -- it reports "0 located" even when the
        #: actor list knows the prop exists. The cone cue survives that because it has a
        #: region-scoped path (`cone_regions` above); without this table the other
        #: families would answer FALSE at the very junction the spawned prop stands in.
        # A plain comma-joined STRING, not a string array. Declaring [""] makes rclpy
        # raise InvalidParameterTypeException against the launch override and the node
        # dies at startup, so the run drives with no cues at all. Same trap the comment
        # on cone_regions describes: launch INFERS types.
        self.declare_parameter("prop_tables", "")

        npz = self.get_parameter("regions_npz").get_parameter_value().string_value
        if not npz:
            npz = self._default_npz()
        self._table = load_region_table(npz) if npz and os.path.exists(npz) else None
        if self._table is None:
            self.get_logger().warn(
                "no region table — 'at an intersection' will be answered from the "
                "cluster id alone, without labels")

        self._cluster: int | None = None
        self._pose: tuple[float, float] | None = None
        #: cone ACTOR ids, from /carla/actor_list (which carries types but no poses)
        self._cone_ids: set[int] = set()
        #: family -> actor ids, same source. Empty until the first actor list arrives.
        self._landmark_ids: dict[str, set[int]] = {}
        self._saw_actor_list = False
        #: spawned region -> landmark family, from the prop tables
        self._region_family: dict[int, str] = self._load_region_families()
        #: family -> regions, for MAP-NATIVE landmarks (traffic lights). Kept SEPARATE from
        #: `_region_family` on purpose: that dict names what this deployment can PLACE, and
        #: the absent-family short-circuit keys off it, so folding lights in there would
        #: make a spawned cone answer traffic_light=False.
        self._native_regions: dict[str, set[int]] = self._load_native_landmarks()
        # PER-WORLD GROUND-TRUTH LANDMARKS. "family:r1,r2;family:r3", e.g.
        # "stop_sign:63,59" adds stop signs at J63 and J59 FOR THIS RUN ONLY, answered by region
        # exactly like the map's own. Nothing is spawned (CARLA has no stop-sign prop), so the
        # camera shows nothing there: this is a ground-truth cue, not a visual one. Logged at
        # startup so a run can prove it was in force.
        self.declare_parameter("extra_landmarks", "")
        extra = str(self.get_parameter("extra_landmarks").value or "").strip()
        for part in [x for x in extra.split(";") if x.strip()]:
            fam, _, regs = part.partition(":")
            self._native_regions.setdefault(fam.strip(), set()).update(int(r) for r in regs.split(",") if r.strip())
        if extra:
            self.get_logger().info(f"extra_landmarks in force: {extra!r} -> "
                                   + ", ".join(f"{f}={sorted(r)}" for f, r in sorted(self._native_regions.items())))
        self.get_logger().info(
            "map-native landmarks: "
            + (", ".join(f"{f} in {len(r)} region(s)"
                         for f, r in sorted(self._native_regions.items()))
               or "NONE -- landmarks.<town>.json missing; traffic-light cues will be "
                  "unanswerable from /carla/objects, which publishes no pose for them"))
        # SAY WHAT WAS LOADED. This mapping decides whether a non-cone landmark can be
        # answered at all, and when it is empty every such cue silently resolves False
        # (a successfully spawned bench still reports bench=False) with nothing on either
        # side of the chain printing why.
        self.get_logger().info(
            f"prop tables: {self.get_parameter('prop_tables').value!r} -> "
            f"{len(self._region_family)} region(s) classified {self._region_family}"
            + ("  [EMPTY -- every non-cone landmark will answer False]"
               if not self._region_family else ""))
        #: id -> (x, y), from /carla/objects (which carries poses but no type strings)
        self._object_xy: dict[int, tuple[float, float]] = {}
        self._warned_no_pose = False
        self._last: dict[str, bool] = {}

        self._pub = self.create_publisher(
            String, self.get_parameter("cue_topic").get_parameter_value().string_value, 10)
        self.create_subscription(
            Int16, "/predicted_cluster", self._cluster_cb, 10)
        self.create_subscription(
            Odometry, self.get_parameter("odom_topic").get_parameter_value().string_value,
            self._odom_cb, 10)
        self._subscribe_actors()

        hz = max(1.0, self.get_parameter("publish_hz").get_parameter_value().double_value)
        self.create_timer(1.0 / hz, self._tick)
        self.get_logger().info(
            f"gt_cue_node: answering cues from ground truth at {hz:.0f} Hz "
            f"(cone radius {self.get_parameter('cone_radius_m').value} m)")

    def _default_npz(self) -> str:
        try:
            from ament_index_python.packages import get_package_share_directory
            p = os.path.join(get_package_share_directory("carla_gt_bridge"),
                             "config", "regions.town05.npz")
            if os.path.exists(p):
                return p
        except Exception:
            pass
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(here, "config", "regions.town05.npz")

    def _subscribe_actors(self) -> None:
        """Subscribe to the actor list AND the object list, and join them on id.

        Neither message is sufficient alone, which is not obvious until you read the
        definitions: ``CarlaActorInfo`` is ``{id, parent_id, type, rolename}`` — it says a
        cone EXISTS but not where — while ``derived_object_msgs/Object`` carries ``pose``
        but no type string. So types come from one and positions from the other, joined on
        the shared id. (Reading a ``transform`` off the actor info raises AttributeError,
        and every cue then silently answers False.)

        Imported lazily and guarded: `carla_msgs` only exists where the bridge is
        installed, and this node is otherwise useful (and testable) without it. Without
        the actor list the cone cue simply answers False, which is the correct answer for
        a world with no cone in it.
        """
        topic = self.get_parameter("actor_list_topic").get_parameter_value().string_value
        try:
            from carla_msgs.msg import CarlaActorList
        except ImportError:
            self.get_logger().warn(
                f"carla_msgs not available — not subscribing to {topic}; "
                f"cone cues will answer False")
            return
        self.create_subscription(CarlaActorList, topic, self._actors_cb, 10)
        try:
            from derived_object_msgs.msg import ObjectArray
            self.create_subscription(ObjectArray, "/carla/objects",
                                     self._objects_cb, 10)
        except ImportError:
            self.get_logger().warn(
                "derived_object_msgs not available — cone cues will fall back to "
                "existence without a distance check")

    # -- callbacks -------------------------------------------------------- #

    def _cluster_cb(self, msg: Int16) -> None:
        self._cluster = int(msg.data)

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self._pose = (float(p.x), float(p.y))

    def _actors_cb(self, msg) -> None:
        self._cone_ids = {
            int(a.id) for a in msg.actors
            if any(c in (a.type or "").lower() for c in CONE_TYPES)}
        # Every other landmark family, same rule. Kept as ids (not a bare bool) so the
        # radius scoping below is identical to the cone's -- "a bench somewhere in the
        # town" must not answer "is there a bench here".
        self._landmark_ids = {
            fam: {int(a.id) for a in msg.actors
                  if any(t in (a.type or "").lower() for t in types)}
            for fam, types in LANDMARK_ACTOR_TYPES.items()}
        self._saw_actor_list = True

    def _objects_cb(self, msg) -> None:
        self._object_xy = {int(o.id): (float(o.pose.position.x),
                                       float(o.pose.position.y))
                           for o in msg.objects}

    def _cone_positions(self) -> list[tuple[float, float]]:
        """Positions of the cones we know about, by joining the two lists on id."""
        return [self._object_xy[i] for i in self._cone_ids if i in self._object_xy]

    # -- the answers ------------------------------------------------------ #

    def _at_intersection(self) -> bool:
        if self._cluster is None:
            return False
        if self._table is None:
            return False
        try:
            return self._table.label_of(self._cluster) == "junction"
        except KeyError:
            return False

    def _forced(self) -> bool | None:
        v = str(self.get_parameter("force_cone").value).strip().lower()
        if v in ("present", "true", "1"):
            return True
        if v in ("absent", "false", "0"):
            return False
        return None

    def _cone_ahead(self) -> bool:
        """A cone near the ego, while the ego is at an intersection.

        Both conditions on purpose. A cone parked elsewhere in the town is not what the
        instruction is about, and answering yes for it would make the branch fire at the
        wrong junction — the same class of error as grounding onto the wrong region.
        """
        forced = self._forced()
        if forced is not None:
            # Still gated on being AT the intersection: a forced answer asserts what the
            # perception would report, not where the vehicle is. Without that gate the
            # branch would fire the moment the plan reaches the decision step, wherever
            # the car happens to be, and the run would prove nothing about placement.
            return forced and self._at_intersection()
        raw = str(self.get_parameter("cone_regions").value or "").strip("[] ")
        regions = [int(x) for x in raw.replace(" ", "").split(",") if x]
        if regions:
            # Ground truth placement: the cone is where we put it, so the answer is
            # region-scoped exactly as the offline oracle's is.
            return self._cluster in regions
        if not self._cone_ids or not self._at_intersection():
            return False
        positions = self._cone_positions()
        if not positions or self._pose is None:
            # A cone exists but we cannot place it — /carla/objects does not list static
            # props in every bridge build. Fall back to existence + at-an-intersection,
            # and SAY so, because it is a weaker claim: any cone in the town now answers
            # yes at any junction.
            if not self._warned_no_pose:
                self.get_logger().warn(
                    f"{len(self._cone_ids)} cone actor(s) known but no pose for any of "
                    f"them on /carla/objects — falling back to existence only, so the "
                    f"radius check is NOT being applied")
                self._warned_no_pose = True
            return True
        r = self.get_parameter("cone_radius_m").get_parameter_value().double_value
        return any(math.dist(self._pose, c) <= r for c in positions)

    def _cone_further_along(self) -> bool | None:
        """Is a known cone ahead of the robot, past the here-and-now radius?

        None when the question cannot be answered -- no pose or no cones located -- so it
        is left out of the answer dict entirely rather than published as a confident
        False. A cue nothing answers must TIME OUT, which is visible; a false negative
        looks like a confident "no cone ahead" and silently sends the robot past the
        junction it was supposed to turn at.
        """
        positions = self._cone_positions()
        if self._pose is None or not positions:
            return None
        import math as _m
        r = self.get_parameter("cone_radius_m").get_parameter_value().double_value
        yaw = self._yaw if getattr(self, "_yaw", None) is not None else None
        for cx, cy in positions:
            d = _m.dist(self._pose, (cx, cy))
            if d <= r:
                continue                      # that is the here-and-now cone, not ahead
            if yaw is None:
                return True                   # cannot check bearing; existence is weaker
            bearing = _m.atan2(cy - self._pose[1], cx - self._pose[0])
            if abs((bearing - yaw + _m.pi) % (2 * _m.pi) - _m.pi) < _m.pi / 3:
                return True                   # within +/-60 deg of straight ahead
        return False

    def _load_native_landmarks(self) -> dict[str, set[int]]:
        """family -> regions, from config/landmarks.<town>.json.

        Its own glob, deliberately not `cones.*`/`props.*`: those feed `_region_family`,
        which is the set of families that can be SPAWNED.
        """
        import json as _json
        out: dict[str, set[int]] = {}
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg = os.path.join(os.path.dirname(here), "config")
        share = os.path.join(here, "config")
        paths = sorted(glob.glob(os.path.join(cfg, "landmarks.*.json"))
                       + glob.glob(os.path.join(share, "landmarks.*.json")))
        for fp in paths:
            try:
                blob = _json.load(open(fp))
            except Exception:                                      # noqa: BLE001
                continue
            for fam, rows in blob.items():
                if fam in ("town", "_note") or not isinstance(rows, list):
                    continue
                out.setdefault(fam, set()).update(
                    int(r["region"]) for r in rows if "region" in r)
        return out

    def _load_region_families(self) -> dict[int, str]:
        """region -> family, read from cones.<town>.json / props.<town>.json.

        The blueprint in each entry's `type` names the family; entries without one are
        cones, which is what the default blueprint has always been.
        """
        import json as _json
        out: dict[int, str] = {}
        #: (file, id, type) for entries that match no family -- reported, never dropped.
        unclassified: list[tuple[str, object, object]] = []
        val = self.get_parameter("prop_tables").value
        paths = ([str(x) for x in val if str(x).strip()]
                 if isinstance(val, (list, tuple))
                 else [x for x in str(val or "").split(",") if x.strip()])
        if not paths:
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cfg = os.path.join(os.path.dirname(here), "config")
            paths = sorted(glob.glob(os.path.join(cfg, "cones.*.json")) +
                           glob.glob(os.path.join(cfg, "props.*.json")))
        for fp in paths:
            if not os.path.exists(fp):
                continue
            try:
                blob = _json.load(open(fp))
            except Exception:                                      # noqa: BLE001
                continue
            for c in (blob.get("cones") or []) + (blob.get("props") or []):
                t = str(c.get("type") or "").lower()
                fam = next((f for f, subs in LANDMARK_ACTOR_TYPES.items()
                            if any(x in t for x in subs)), None)
                if fam is None:
                    fam = "cone" if not t or "cone" in t else None
                if fam:
                    out[int(c["region"])] = fam
                else:
                    # LOUD, NOT DROPPED. If an entry whose blueprint matches no family
                    # were dropped silently, the prop would SPAWN, the actor list would
                    # show it, and the cue for it would answer False at the very region
                    # it is standing in -- so the run completes and reads as "the plan
                    # ignored the world".
                    unclassified.append((fp, c.get("id"), c.get("type")))
        if unclassified:
            detail = "; ".join(f"{i} type={ty!r} in {os.path.basename(f)}"
                               for f, i, ty in unclassified)
            self.get_logger().error(
                f"{len(unclassified)} prop-table entr(ies) match NO landmark family and "
                f"will answer False at their own region: {detail}. Add the blueprint "
                f"substring to LANDMARK_ACTOR_TYPES and the cue spellings to "
                f"cue_answers.LANDMARK_SPELLINGS.")
        return out

    def _spawned_regions(self) -> list[int]:
        raw = str(self.get_parameter("cone_regions").value or "").strip("[] ")
        return [int(x) for x in raw.replace(" ", "").split(",") if x]

    def _landmark_near(self, fam: str) -> bool | None:
        """Is a landmark of ``fam`` within cone_radius_m of the ego?

        None means UNANSWERABLE, and the distinction matters: before the actor list has
        arrived we know nothing, and reporting False would make a branch take its default
        on a confident-looking negative. Once the list HAS arrived, a family with no
        actors in the town is a real False -- the town genuinely has none.
        """
        # REGION-SCOPED GROUND TRUTH FIRST, exactly as the cone cue does it. We placed the
        # prop, so we know where it is; asking /carla/objects for a pose it never publishes
        # turns a known-true answer into a false one.
        native = self._native_regions.get(fam)
        if native is not None:
            # MAP-NATIVE: in every world, never spawned, and `/carla/objects` publishes no
            # pose for it even though the actor list knows it exists. Proximity therefore
            # CANNOT be computed downstream, so this region-scoped answer is the only one
            # available. Same shape as the cone cue.
            if self._cluster is None:
                return None
            return int(self._cluster) in native

        spawned = self._spawned_regions()
        mine = [r for r in spawned if self._region_family.get(r) == fam]
        if mine:
            return self._cluster in mine
        if spawned and self._region_family and fam in set(self._region_family.values()):
            # Something was spawned and none of it is this family, so the family is
            # genuinely absent from this world -- a real False, not an unknown.
            #
            # ONLY FOR A FAMILY THIS DEPLOYMENT CAN PLACE. A family that lives in the MAP
            # -- traffic lights, stop signs -- is not absent merely because we spawned
            # something else; answering False for it here would skip the actor-list
            # lookup below, where those actors actually are (e.g. a mission that spawns a
            # cone could then never fire a Detect(TrafficLight) branch).
            return False
        if not self._saw_actor_list:
            return None
        ids = self._landmark_ids.get(fam) or set()
        if not ids:
            return False
        if self._pose is None:
            return None
        r = float(self.get_parameter("cone_radius_m").value)
        ex, ey = self._pose
        for i in ids:
            xy = self._object_xy.get(i)
            if xy is None:
                continue
            if math.hypot(xy[0] - ex, xy[1] - ey) <= r:
                return True
        return False

    def _tick(self) -> None:
        self._place_first = bool(self.get_parameter("cue_place_first").value)
        at_junction = self._at_intersection()
        cone = self._cone_ahead()
        # THE VOCABULARY LIVES IN ONE PLACE. This node and its offline twin
        # (`scripts/missions.py::cue_oracle`) both delegate to cue_answers, so the
        # offline harness cannot certify plans the robot then executes differently.
        # LOOKAHEAD. `cone` above is "a cone is within radius of me NOW". A mission like
        # "turn at the intersection BEFORE the one with the cone" needs a different
        # question -- "is there a cone at the next junction" -- which a forward-looking
        # camera can answer from here and the region classifier cannot. Ground truth
        # stands in for that camera so the PLAN side is testable before the perception
        # side exists: any known cone that is ahead of the robot and beyond the local
        # radius counts as being at a junction further along.
        cone_ahead = self._cone_further_along()
        landmarks = {fam: self._landmark_near(fam) for fam in LANDMARK_ACTOR_TYPES}
        landmarks = {k: v for k, v in landmarks.items() if v is not None}
        answers = cue_answers.answers_for_world(
            at_junction, cone, place_first=self._place_first,
            cone_ahead=cone_ahead, landmarks=landmarks)
        if answers != self._last:
            # LOG THE FAMILIES, not just the two summary booleans: a log that shows only
            # `at_intersection` and `cone` while ~20 keys go on the wire invites wrong
            # conclusions about which families were answered. A family that is
            # UNANSWERABLE is the one thing worth seeing here, because brain then waits
            # the full cue timeout and the run looks like a driving failure.
            fam = {f: self._landmark_near(f) for f in LANDMARK_ACTOR_TYPES}
            shown = " ".join(
                f"{f}={'?' if v is None else v}" for f, v in sorted(fam.items()))
            unanswerable = sorted(f for f, v in fam.items() if v is None)
            self.get_logger().info(
                f"cue answers changed: at_intersection={at_junction} cone={cone} "
                f"| {shown} | {len(answers)} keys published"
                + (f" | UNANSWERABLE: {','.join(unanswerable)}" if unanswerable else "")
                + f" (cluster {self._cluster}, {len(self._cone_ids)} cone actors known, "
                f"{len(self._cone_positions())} located)")
            self._last = dict(answers)
        self._pub.publish(String(data=json.dumps(answers)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GtCueNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
