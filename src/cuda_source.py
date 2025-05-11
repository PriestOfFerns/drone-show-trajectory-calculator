CUDA_KERNEL_SOURCE = """
#include <math.h> // For sqrtf, fmaxf, fminf, fabsf

extern "C" {

// Constant for geometric calculations (e.g., checking for parallel lines, zero length segments)
#define GEOMETRIC_EPSILON 1e-7f
// Constant for the final intersection check (comparing distance to sum of radii)
#define INTERSECTION_EPSILON 1e-8f

// Helper function: Squared distance between two 3D points
__device__ float distSq(float p1x, float p1y, float p1z, float p2x, float p2y, float p2z) {
    float dx = p1x - p2x;
    float dy = p1y - p2y;
    float dz = p1z - p2z;
    return dx*dx + dy*dy + dz*dz;
}

// Helper function: Dot product of two 3D vectors
__device__ float dot_product(float v1x, float v1y, float v1z, float v2x, float v2y, float v2z) {
    return v1x * v2x + v1y * v2y + v1z * v2z;
}

// Calculates the shortest squared distance between two line segments in 3D space.
// Based on the logic from 'closest_points_on_segments' in capsule.py.
__device__ float closest_squared_distance_between_segments(
    float p1x, float p1y, float p1z, float q1x, float q1y, float q1z, // Segment 1 (from p1 to q1)
    float p2x, float p2y, float p2z, float q2x, float q2y, float q2z  // Segment 2 (from p2 to q2)
) {
    // Directions of the segments
    float d1x = q1x - p1x;
    float d1y = q1y - p1y;
    float d1z = q1z - p1z;

    float d2x = q2x - p2x;
    float d2y = q2y - p2y;
    float d2z = q2z - p2z;

    // Vector between the start points of the segments
    float rx = p1x - p2x;
    float ry = p1y - p2y;
    float rz = p1z - p2z;

    float a = dot_product(d1x, d1y, d1z, d1x, d1y, d1z); // Squared length of segment 1
    float e = dot_product(d2x, d2y, d2z, d2x, d2y, d2z); // Squared length of segment 2
    float f = dot_product(d2x, d2y, d2z, rx, ry, rz);

    float c1_final_x, c1_final_y, c1_final_z; // Closest point on segment 1
    float c2_final_x, c2_final_y, c2_final_z; // Closest point on segment 2

    // Check if either segment is a point (or very short)
    if (a <= GEOMETRIC_EPSILON && e <= GEOMETRIC_EPSILON) { // Both segments are points
        c1_final_x = p1x; c1_final_y = p1y; c1_final_z = p1z;
        c2_final_x = p2x; c2_final_y = p2y; c2_final_z = p2z;
        return distSq(c1_final_x, c1_final_y, c1_final_z, c2_final_x, c2_final_y, c2_final_z);
    }

    if (a <= GEOMETRIC_EPSILON) { // First segment is a point
        c1_final_x = p1x; c1_final_y = p1y; c1_final_z = p1z;
        // Project point p1 onto the second segment (p2, q2)
        float t_s2 = dot_product(p1x - p2x, p1y - p2y, p1z - p2z, d2x, d2y, d2z) / e;
        t_s2 = fmaxf(0.0f, fminf(1.0f, t_s2)); // Clamp t_s2 to [0, 1]
        c2_final_x = p2x + t_s2 * d2x;
        c2_final_y = p2y + t_s2 * d2y;
        c2_final_z = p2z + t_s2 * d2z;
        return distSq(c1_final_x, c1_final_y, c1_final_z, c2_final_x, c2_final_y, c2_final_z);
    }

    if (e <= GEOMETRIC_EPSILON) { // Second segment is a point
        c2_final_x = p2x; c2_final_y = p2y; c2_final_z = p2z;
        // Project point p2 onto the first segment (p1, q1)
        float t_s1 = dot_product(p2x - p1x, p2y - p1y, p2z - p1z, d1x, d1y, d1z) / a;
        t_s1 = fmaxf(0.0f, fminf(1.0f, t_s1)); // Clamp t_s1 to [0, 1]
        c1_final_x = p1x + t_s1 * d1x;
        c1_final_y = p1y + t_s1 * d1y;
        c1_final_z = p1z + t_s1 * d1z;
        return distSq(c1_final_x, c1_final_y, c1_final_z, c2_final_x, c2_final_y, c2_final_z);
    }

    // General case: both segments are not points
    float c_dot_d1_r = dot_product(d1x, d1y, d1z, rx, ry, rz);
    float b_dot_d1_d2 = dot_product(d1x, d1y, d1z, d2x, d2y, d2z);
    float denom = a * e - b_dot_d1_d2 * b_dot_d1_d2;

    float s, t; // Parameters for the closest points on the lines defined by segments

    // If segments are parallel or nearly parallel
    if (fabsf(denom) < GEOMETRIC_EPSILON) {
        t = -c_dot_d1_r / a; // Equivalent to np.dot(d1, -r) / a
        t = fmaxf(0.0f, fminf(1.0f, t)); // Clamp t

        // Closest point on segment 1 based on clamped t
        float temp_c1x = p1x + t * d1x;
        float temp_c1y = p1y + t * d1y;
        float temp_c1z = p1z + t * d1z;

        // Project this point onto segment 2
        s = dot_product(temp_c1x - p2x, temp_c1y - p2y, temp_c1z - p2z, d2x, d2y, d2z) / e;
        s = fmaxf(0.0f, fminf(1.0f, s)); // Clamp s
    } else {
        // Non-parallel segments
        float inv_denom = 1.0f / denom;
        t = (b_dot_d1_d2 * f - c_dot_d1_r * e) * inv_denom;
        s = (a * f - b_dot_d1_d2 * c_dot_d1_r) * inv_denom;

        float t_clamped = fmaxf(0.0f, fminf(1.0f, t));
        float s_clamped = fmaxf(0.0f, fminf(1.0f, s));

        // Refined clamping logic as in capsule.py
        if (fabsf(t_clamped - t) > GEOMETRIC_EPSILON) { // If t was clamped
            // Recalculate s based on the clamped t
            float c1_on_line_x = p1x + t_clamped * d1x;
            float c1_on_line_y = p1y + t_clamped * d1y;
            float c1_on_line_z = p1z + t_clamped * d1z;
            float s_recalculated = dot_product(
                c1_on_line_x - p2x, c1_on_line_y - p2y, c1_on_line_z - p2z,
                d2x, d2y, d2z) / e;
            s_clamped = fmaxf(0.0f, fminf(1.0f, s_recalculated));
        } else if (fabsf(s_clamped - s) > GEOMETRIC_EPSILON) { // Else if s was clamped (and t was not)
            // Recalculate t based on the clamped s
            float c2_on_line_x = p2x + s_clamped * d2x;
            float c2_on_line_y = p2y + s_clamped * d2y;
            float c2_on_line_z = p2z + s_clamped * d2z;
            float t_recalculated = dot_product(
                c2_on_line_x - p1x, c2_on_line_y - p1y, c2_on_line_z - p1z,
                d1x, d1y, d1z) / a;
            t_clamped = fmaxf(0.0f, fminf(1.0f, t_recalculated));
        }
        t = t_clamped;
        s = s_clamped;
    }

    // Calculate the closest points on the segments using the final s and t
    c1_final_x = p1x + t * d1x;
    c1_final_y = p1y + t * d1y;
    c1_final_z = p1z + t * d1z;

    c2_final_x = p2x + s * d2x;
    c2_final_y = p2y + s * d2y;
    c2_final_z = p2z + s * d2z;

    return distSq(c1_final_x, c1_final_y, c1_final_z, c2_final_x, c2_final_y, c2_final_z);
}


// Checks if two capsules are intersecting.
// Replaces the placeholder 'are_capsules_intersecting_simplified'.
__device__ bool are_capsules_intersecting_kernel_logic(
    float s1p1x, float s1p1y, float s1p1z, float s1p2x, float s1p2y, float s1p2z, float r1, // Capsule 1
    float s2p1x, float s2p1y, float s2p1z, float s2p2x, float s2p2y, float s2p2z, float r2  // Capsule 2
) {
    // Calculate the shortest squared distance between the core line segments
    float squared_dist_segments = closest_squared_distance_between_segments(
        s1p1x, s1p1y, s1p1z, s1p2x, s1p2y, s1p2z,
        s2p1x, s2p1y, s2p1z, s2p2x, s2p2y, s2p2z
    );

    float distance_segments = sqrtf(squared_dist_segments);
    float combined_radius = r1 + r2;

    // Capsules intersect if the distance between their core segments is less than or equal
    // to the sum of their radii (plus a small epsilon for floating point comparisons).
    return distance_segments <= (combined_radius + INTERSECTION_EPSILON);
}


__global__ void count_intersections_kernel(
    const float* self_capsules_data, int num_self_capsules,
    const float* other_capsules_data, int num_other_capsules,
    unsigned int* collision_counts_out) // Output array (size 1)
{
    int self_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (self_idx < num_self_capsules) {
        // Extract data for the current "self" capsule
        // Each capsule: p1(x,y,z), p2(x,y,z), radius = 7 floats
        float s1p1x = self_capsules_data[self_idx * 7 + 0];
        float s1p1y = self_capsules_data[self_idx * 7 + 1];
        float s1p1z = self_capsules_data[self_idx * 7 + 2];
        float s1p2x = self_capsules_data[self_idx * 7 + 3];
        float s1p2y = self_capsules_data[self_idx * 7 + 4];
        float s1p2z = self_capsules_data[self_idx * 7 + 5];
        float r1    = self_capsules_data[self_idx * 7 + 6];

        unsigned int local_collisions = 0;
        // Iterate through all "other" capsules
        for (int other_idx = 0; other_idx < num_other_capsules; ++other_idx) {
            float s2p1x = other_capsules_data[other_idx * 7 + 0];
            float s2p1y = other_capsules_data[other_idx * 7 + 1];
            float s2p1z = other_capsules_data[other_idx * 7 + 2];
            float s2p2x = other_capsules_data[other_idx * 7 + 3];
            float s2p2y = other_capsules_data[other_idx * 7 + 4];
            float s2p2z = other_capsules_data[other_idx * 7 + 5];
            float r2    = other_capsules_data[other_idx * 7 + 6];

            // Call the device function to check for intersection
            if (are_capsules_intersecting_kernel_logic(
                    s1p1x, s1p1y, s1p1z, s1p2x, s1p2y, s1p2z, r1,
                    s2p1x, s2p1y, s2p1z, s2p2x, s2p2y, s2p2z, r2)) {
                local_collisions++;
            }
        }
        // Atomically add the count of collisions found by this thread to the global counter
        if (local_collisions > 0) {
            atomicAdd(&collision_counts_out[0], local_collisions);
        }
    }
}
} // extern "C"
"""

