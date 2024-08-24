#include "aruco_tracker.h"
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include "ceres/ceres.h"

ArucoProcessor::ArucoProcessor(const std::string &aruco_dict_name, double marker_length, const std::map<int, Eigen::Vector3d> &marker_positions, const std::vector<Camera> &camera_params)
{
}


struct TagObservation;

class ArucoProcessor {
public:
    std::pair<Eigen::Vector3d, Eigen::Vector3d> optimizePose(
        const Eigen::VectorXd &initial_guess,
        const std::vector<TagObservation> &observations
    ) {
        // Define a Ceres problem
        ceres::Problem problem;

        // Split the initial guess into position and orientation
        Eigen::Vector3d position = initial_guess.head<3>();
        Eigen::Vector3d orientation = initial_guess.tail<3>();

        // Iterate over all the observations
        for (const auto &obs : observations) {
            // Create a cost function using a lambda that captures the necessary variables
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<ReprojectionCostFunctor, 2, 3, 3>(
                    new ReprojectionCostFunctor(obs, this->marker_positions_[obs.marker_id], obs.cam));

            // Add the cost function to the problem
            problem.AddResidualBlock(cost_function, nullptr, position.data(), orientation.data());
        }

        // Configure the solver options
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;

        // Solve the problem
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // Return the optimized position and orientation
        return std::make_pair(position, orientation);
    }

private:
    std::map<int, Eigen::Vector3d> marker_positions_;
};

// Functor for reprojection error
struct ReprojectionCostFunctor {
    ReprojectionCostFunctor(const TagObservation& obs, const Eigen::Vector3d& marker_position, const Camera& cam)
        : obs_(obs), marker_position_(marker_position), cam_(cam) {}

    template <typename T>
    bool operator()(const T* const position, const T* const orientation, T* residuals) const {
        // Convert orientation vector to rotation matrix
        Eigen::Matrix<T, 3, 3> rotation_matrix;
        // Your method of conversion here, possibly using ceres::AngleAxisToRotationMatrix
        Eigen::AngleAxis<T> rotation(Eigen::Matrix<T, 3, 1>(orientation[0], orientation[1], orientation[2]));
        rotation_matrix = rotation.toRotationMatrix();

        // Transform marker position into the camera frame
        Eigen::Matrix<T, 3, 1> transformed_position = rotation_matrix * marker_position_.cast<T>() + Eigen::Matrix<T, 3, 1>(position[0], position[1], position[2]);

        // Project the transformed 3D position onto the 2D image plane
        T predicted_x = (cam_.camera_matrix.at<double>(0, 0) * transformed_position.x() + cam_.camera_matrix.at<double>(0, 2)) / transformed_position.z();
        T predicted_y = (cam_.camera_matrix.at<double>(1, 1) * transformed_position.y() + cam_.camera_matrix.at<double>(1, 2)) / transformed_position.z();

        // Compute residuals (difference between observed and predicted points)
        residuals[0] = T(obs_.corners[0].x) - predicted_x;
        residuals[1] = T(obs_.corners[0].y) - predicted_y;

        return true;
    }

    const TagObservation& obs_;
    const Eigen::Vector3d marker_position_;
    const Camera& cam_;
};