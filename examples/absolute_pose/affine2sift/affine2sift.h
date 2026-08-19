#include <Eigen/Dense>
#include <numeric>
#include <vector>

using namespace Eigen;

// Converts an affine correspondence to a scale-and-orientation correspondence.
// There are eight possible solutions, but this arbitrarily returns the first one.
void affine2sift(const Eigen::Matrix2d &A,         // input affine transformation matrix
                 double &s_ref, double &c_ref,     // sine and cosine of feature orientation in reference image
                 double &s_query, double &c_query, // sine and cosine of feature orientation in query image
                 double &q // ratio of feature scales (scale in query image / scale in reference image)
);
