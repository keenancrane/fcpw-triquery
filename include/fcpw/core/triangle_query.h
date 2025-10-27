#pragma once

#include <fcpw/core/core.h>
#include <fcpw/core/bounding_volumes.h>

namespace fcpw {

template<size_t DIM>
struct TriangleQuery;

template<>
struct TriangleQuery<3> {
    // constructor
    TriangleQuery() : a(Vector3::Zero()), b(Vector3::Zero()), c(Vector3::Zero()), r2(maxFloat) {}
    TriangleQuery(const Vector3& a_, const Vector3& b_, const Vector3& c_, float r2_=maxFloat)
    : a(a_), b(b_), c(c_), r2(r2_) {}

    // centroid
    inline Vector3 centroid() const {
        return (a + b + c)/3.0f;
    }

    // computes transformed triangle query
    inline TriangleQuery<3> transform(const Transform<3>& t) const {
        Vector3 ta = t*a;
        Vector3 tb = t*b;
        Vector3 tc = t*c;

        float tr2 = maxFloat;
        if (r2 < maxFloat) {
            // approximate scaling of the distance bound using a unit direction from the centroid
            Vector3 q = centroid();
            Vector3 dir = Vector3::Zero();
            dir[0] = 1.0f; // heuristic consistent with BoundingSphere transform
            float r = std::sqrt(r2);
            float scaled = (t*(q + dir*r) - t*q).squaredNorm();
            tr2 = scaled;
        }

        return TriangleQuery<3>(ta, tb, tc, tr2);
    }

    // members
    Vector3 a, b, c;
    float r2; // current best squared distance upper bound
};

} // namespace fcpw


