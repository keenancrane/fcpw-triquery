#include <fcpw/fcpw.h>
#include <iostream>
#include <cmath>

using namespace fcpw;

static bool approxEqual(float a, float b, float eps=1e-5f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    // Build a simple scene: single mesh triangle on z=0
    Scene<3> scene;
    scene.setObjectCount(1);

    std::vector<Vector<3>> positions = {
        Vector3(0.0f, 0.0f, 0.0f),
        Vector3(1.0f, 0.0f, 0.0f),
        Vector3(0.0f, 1.0f, 0.0f)
    };
    std::vector<Vector3i> indices = { Vector3i(0, 1, 2) };

    scene.setObjectVertices(positions, 0);
    scene.setObjectTriangles(indices, 0);
    scene.build(AggregateType::Bvh_SurfaceArea, false);

    int failures = 0;

    // Case 1: identical triangles (distance 0)
    {
        Interaction<3> i;
        bool found = scene.findClosestPointToTriangle(positions[0], positions[1], positions[2], i, nullptr, maxFloat, true);
        if (!found || !approxEqual(i.d, 0.0f)) {
            std::cerr << "Case 1 failed: expected distance 0, got " << i.d << std::endl;
            failures++;
        }
        // Normal should be +/- (0,0,1)
        if (i.n.norm() > 0.0f) {
            Vector3 nExpected(0.0f, 0.0f, 1.0f);
            if (!approxEqual(std::fabs(i.n.dot(nExpected)), 1.0f, 1e-4f)) {
                std::cerr << "Case 1 warning: normal unexpected: " << i.n.transpose() << std::endl;
            }
        }
    }

    // Case 2: parallel triangle at z=2 (distance 2)
    {
        Vector3 a(0.0f, 0.0f, 2.0f), b(1.0f, 0.0f, 2.0f), c(0.0f, 1.0f, 2.0f);
        Interaction<3> i;
        Vector3 q;
        bool found = scene.findClosestPointToTriangle(a, b, c, i, &q, maxFloat, true);
        if (!found || !approxEqual(i.d, 2.0f)) {
            std::cerr << "Case 2 failed: expected distance 2, got " << i.d << std::endl;
            failures++;
        }
    }

    // Case 3: intersecting triangles (share an edge) -> distance 0
    {
        Vector3 a(0.0f, 0.0f, 0.0f), b(1.0f, 0.0f, 0.0f), c(0.5f, -0.5f, 0.0f);
        Interaction<3> i;
        bool found = scene.findClosestPointToTriangle(a, b, c, i);
        if (!found || !approxEqual(i.d, 0.0f)) {
            std::cerr << "Case 3 failed: expected distance 0, got " << i.d << std::endl;
            failures++;
        }
    }

    // Case 4: disjoint non-parallel (edge-edge closest)
    {
        Vector3 a(0.5f, -0.5f, 1.0f), b(1.5f, -0.5f, 1.5f), c(0.5f, 0.5f, 2.0f);
        Interaction<3> i;
        bool found = scene.findClosestPointToTriangle(a, b, c, i);
        if (!found || !(i.d >= 0.0f)) {
            std::cerr << "Case 4 failed: expected non-negative distance, got " << i.d << std::endl;
            failures++;
        }
    }

    if (failures == 0) {
        std::cout << "Triangle query tests passed." << std::endl;
        return EXIT_SUCCESS;
    }

    return EXIT_FAILURE;
}


