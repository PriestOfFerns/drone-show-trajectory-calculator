import numpy as np


class Capsule:
    """
    Represents a capsule shape defined by a line segment and a radius.

    Attributes:
        p1 (np.ndarray): The first endpoint of the capsule's core line segment.
        p2 (np.ndarray): The second endpoint of the capsule's core line segment.
        radius (float): The radius of the capsule.
    """

    def __init__(self, p1, p2, radius):
        # Ensure p1 and p2 are numpy arrays for easier vector math
        self.p1 = np.asarray(p1, dtype=float)
        self.p2 = np.asarray(p2, dtype=float)
        self.radius = float(radius)


def closest_points_on_segments(p1, q1, p2, q2):
    """
    Calculates the shortest distance between two line segments in 3D space
    and returns the closest points on each segment.

    Args:
        p1 (np.ndarray): Start point of the first segment.
        q1 (np.ndarray): End point of the first segment.
        p2 (np.ndarray): Start point of the second segment.
        q2 (np.ndarray): End point of the second segment.

    Returns:
        tuple: A tuple containing:
            - distance (float): The shortest distance between the segments.
            - closest_point1 (np.ndarray): The closest point on the first segment.
            - closest_point2 (np.ndarray): The closest point on the second segment.
    """
    # Directions of the segments
    d1 = q1 - p1
    d2 = q2 - p2

    # Vector between the start points
    r = p1 - p2

    # Dot products
    a = np.dot(d1, d1)  # Squared length of segment 1
    e = np.dot(d2, d2)  # Squared length of segment 2
    f = np.dot(d2, r)

    # Check if either segment is a point
    epsilon = 1e-8  # Small tolerance for floating point comparisons

    if a <= epsilon and e <= epsilon:
        # Both segments are points
        closest_point1 = p1
        closest_point2 = p2
        distance = np.linalg.norm(closest_point1 - closest_point2)
        return distance, closest_point1, closest_point2

    if a <= epsilon:
        # First segment is a point
        closest_point1 = p1
        # Project point p1 onto the second segment
        s = np.dot(d2, r) / e
        s = np.clip(s, 0.0, 1.0)  # Clamp s to [0, 1]
        closest_point2 = p2 + s * d2
        distance = np.linalg.norm(closest_point1 - closest_point2)
        return distance, closest_point1, closest_point2

    if e <= epsilon:
        # Second segment is a point
        closest_point2 = p2
        # Project point p2 onto the first segment
        t = np.dot(d1, -r) / a
        t = np.clip(t, 0.0, 1.0)  # Clamp t to [0, 1]
        closest_point1 = p1 + t * d1
        distance = np.linalg.norm(closest_point1 - closest_point2)
        return distance, closest_point1, closest_point2

    # General case: both segments are lines or line segments
    c = np.dot(d1, r)
    b = np.dot(d1, d2)
    denom = a * e - b * b  # Denominator for s and t

    # If segments are parallel or nearly parallel
    if abs(denom) < epsilon:
        # Choose an arbitrary point on one segment (e.g., p1) and project it
        # onto the other segment. Clamp the projection parameter to [0, 1].
        t = np.dot(d1, -r) / a
        t = np.clip(t, 0.0, 1.0)
        closest_point1 = p1 + t * d1
        # Project closest_point1 onto the second segment
        s = np.dot(d2, closest_point1 - p2) / e
        s = np.clip(s, 0.0, 1.0)
        closest_point2 = p2 + s * d2
    else:
        # Non-parallel segments
        inv_denom = 1.0 / denom
        # Calculate parameters t and s for the closest points
        t = (b * f - c * e) * inv_denom
        s = (a * f - b * c) * inv_denom

        # Clamp t and s to the [0, 1] range to find the closest points on the segments
        t_clamped = np.clip(t, 0.0, 1.0)
        s_clamped = np.clip(s, 0.0, 1.0)

        # Check if clamping occurred for t
        if abs(t_clamped - t) > epsilon:
            # If t was clamped, recalculate s based on the clamped t
            s_recalculated = np.dot(d2, (p1 + t_clamped * d1) - p2) / e
            s_clamped = np.clip(s_recalculated, 0.0, 1.0)
        # Check if clamping occurred for s (and if t wasn't clamped in the previous step)
        elif abs(s_clamped - s) > epsilon:
            # If s was clamped, recalculate t based on the clamped s
            t_recalculated = np.dot(d1, (p2 + s_clamped * d2) - p1) / a
            t_clamped = np.clip(t_recalculated, 0.0, 1.0)

        # Calculate the closest points using the clamped parameters
        closest_point1 = p1 + t_clamped * d1
        closest_point2 = p2 + s_clamped * d2

    # Calculate the distance between the closest points
    distance = np.linalg.norm(closest_point1 - closest_point2)

    return distance, closest_point1, closest_point2


def closest_point_on_segment_to_point(segment_p1, segment_p2, point):
    """
    Calculates the closest point on a line segment to a given point.

    Args:
        segment_p1 (np.ndarray): The start point of the line segment.
        segment_p2 (np.ndarray): The end point of the line segment.
        point (np.ndarray): The point to find the closest point on the segment to.

    Returns:
        np.ndarray: The closest point on the segment to the given point.
    """
    segment_direction = segment_p2 - segment_p1
    segment_length_sq = np.dot(segment_direction, segment_direction)

    # If the segment is a point
    epsilon = 1e-8
    if segment_length_sq < epsilon:
        return segment_p1

    # Project the point onto the line containing the segment
    t = np.dot(point - segment_p1, segment_direction) / segment_length_sq

    # Clamp the projection parameter t to the [0, 1] range
    t_clamped = np.clip(t, 0.0, 1.0)

    # Calculate the closest point on the segment
    closest_pt = segment_p1 + t_clamped * segment_direction

    return closest_pt


def are_capsules_intersecting(capsule1, capsule2):
    """
    Checks if two Capsule objects are intersecting.

    Args:
        capsule1 (Capsule): The first capsule.
        capsule2 (Capsule): The second capsule.

    Returns:
        bool: True if the capsules are intersecting, False otherwise.
    """
    # Calculate the shortest distance between the core line segments
    distance_between_segments, _, _ = closest_points_on_segments(
        capsule1.p1, capsule1.p2, capsule2.p1, capsule2.p2
    )

    # Capsules intersect if the distance between segments is less than or equal
    # to the sum of their radii. Use a small tolerance for floating point comparisons.
    return distance_between_segments <= (capsule1.radius + capsule2.radius + 1e-8)


def are_capsules_intersecting(capsule1, capsule2):
    """
    Checks if two Capsule objects are intersecting.

    Args:
        capsule1 (Capsule): The first capsule.
        capsule2 (Capsule): The second capsule.

    Returns:
        bool: True if the capsules are intersecting, False otherwise.
    """
    # Calculate the shortest distance between the core line segments
    distance_between_segments, _, _ = closest_points_on_segments(
        capsule1.p1, capsule1.p2, capsule2.p1, capsule2.p2
    )

    # Capsules intersect if the distance between segments is less than or equal
    # to the sum of their radii. Use a small tolerance for floating point comparisons.
    return distance_between_segments <= (capsule1.radius + capsule2.radius + 1e-8)


def are_sphere_capsule_intersecting(sphere_center, sphere_radius, capsule):
    """
    Checks if a sphere and a capsule are intersecting.

    Args:
        sphere_center (np.ndarray): The center of the sphere.
        sphere_radius (float): The radius of the sphere.
        capsule (Capsule): The capsule object.

    Returns:
        bool: True if the sphere and capsule are intersecting, False otherwise.
    """
    # Find the closest point on the capsule's core line segment to the sphere's center
    closest_pt_on_segment = closest_point_on_segment_to_point(
        capsule.p1, capsule.p2, sphere_center
    )

    # Calculate the distance between the sphere's center and the closest point on the segment
    distance = np.linalg.norm(sphere_center - closest_pt_on_segment)

    # The sphere and capsule intersect if the distance is less than or equal
    # to the sum of their radii. Use a small tolerance for floating point comparisons.
    return distance <= (sphere_radius + capsule.radius + 1e-8)
