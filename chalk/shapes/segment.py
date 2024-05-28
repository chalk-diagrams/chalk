# from __future__ import annotations

# import math
# from dataclasses import dataclass, field
# from typing import TYPE_CHECKING, Any, List, Optional, Tuple

# import chalk.transform as tx
# from chalk.envelope import Envelope
# from chalk.trace import Trace
# from chalk.transform import P2_t, V2_t, Scalar
# from chalk.types import Enveloped, Traceable, TrailLike
# import numpy as np

# if TYPE_CHECKING:
#     from chalk.trail import Trail

# SignedDistance = Scalar

# Ident = tx.ident
# ORIGIN = tx.P2(0, 0)


# @dataclass
# class LocatedSegment(Traceable, Enveloped, tx.Transformable, TrailLike):
#     offset: V2_t
#     origin: P2_t = field(default_factory=lambda: ORIGIN)

#     @property
#     def p(self) -> P2_t:
#         return self.origin

#     @staticmethod
#     def from_points(p: P2_t, q: P2_t) -> LocatedSegment:
#         return LocatedSegment(q - p, p)

#     def apply_transform(self, t: tx.Affine) -> LocatedSegment:
#         return LocatedSegment(
#             tx.apply_affine(t, self.offset), tx.apply_affine(t, self.origin)
#         )

#     @property
#     def q(self) -> P2_t:
#         return self.p + self.offset

#     def get_trace(self, t: tx.Affine = Ident) -> Trace:
#         def f(point: P2_t, direction: V2_t) -> List[Scalar]:
#             ray = (point, direction)
#             inter = sorted(line_segment(ray, self))
#             return [r / tx.length(direction) for r in inter]

#         return Trace(f)

#     def get_envelope(self, t: tx.Affine = Ident) -> Envelope:
#         def f(d: V2_t) -> SignedDistance:
#             return tx.np.maximum(d @ self.q, d @ self.p)

#         return Envelope(f)

#     @property
#     def length(self) -> Any:
#         return tx.length(self.offset)

#     def to_ray(self) -> Tuple[P2_t, V2_t]:
#         return (self.p, self.q - self.p)

#     def to_trail(self) -> Trail:
#         from chalk.trail import Trail

#         return Trail([Segment(self.offset)])


# @dataclass
# class Segment(LocatedSegment, TrailLike):
#     @property
#     def p(self) -> P2_t:
#         return ORIGIN

#     def apply_transform(self, t: tx.Affine) -> Segment:
#         return Segment(tx.apply_affine(tx.remove_translation(t), self.offset) )

#     def reverse(self):  # type: ignore
#         return self.scale(-1)


# # def seg(offset: V2_t) -> Trail:
# #     return Segment(offset).to_trail()




# def line_segment(ray: Tuple[P2_t, V2_t], segment: LocatedSegment) -> List[float]:
#     """Given a ray and a segment, return the parameter `t` for which the ray
#     meets the segment, that is:

#     ray t₁ = segment.to_ray t₂, with t₂ ∈ [0, segment.length]

#     Note: We need to consider the segment's length separately since `Ray`
#     normalizes the direction to unit and hences looses this information. The
#     length is important to determine whether the intersection point falls
#     within the given segment.

#     See also: https://github.com/danoneata/chalk/issues/91

#     """
#     ray_s = segment.to_ray()
#     t = ray_ray_intersection(ray, ray_s)
#     if not t:
#         return []
#     else:
#         t1, t2 = t
#         # the intersection point is given by any of the two expressions:
#         # ray.anchor   + t1 * ray.direction
#         # ray_s.anchor + t2 * ray_s.direction
#         if 0 <= t2 <= segment.length:
#             # intersection point is in segment
#             return [t1]
#         else:
#             # intersection point outside
#             return []
