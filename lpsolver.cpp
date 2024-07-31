#include <iostream>
#include <limits>
#include <vector>
#include "../HiGHS/src/Highs.h"

extern "C" {
struct LPResult {
    int success;
    double* x;
    double fun;

    LPResult() : success(0), x(nullptr), fun(0.0) {}
    ~LPResult() { delete[] x; }
};

LPResult* solve_lp(int dims, double* c, double* A_ub, double* b_ub, double* bounds_lower, double* bounds_upper) {
    std::cout << "Entering solve_lp" << std::endl;
    std::cout << "Number of variables (dims): " << dims << std::endl;

    auto result = new LPResult();
    result->x = new double[dims];

    Highs highs;
    highs.setOptionValue("output_flag", true);  // Enable output for library-level debugging

    // Log and add variables
    for (int i = 0; i < dims; ++i) {
        std::cout << "Preparing to add Column " << i << ": ";
        std::cout << "Cost = " << c[i] << ", Lower Bound = " << bounds_lower[i] << ", Upper Bound = " << bounds_upper[i] << std::endl;

        if (bounds_lower[i] > bounds_upper[i]) {
            std::cout << "Invalid bounds detected for column " << i << std::endl;
            continue;
        }

        highs.addCol(c[i], bounds_lower[i], bounds_upper[i], 0, nullptr, nullptr);
        std::cout << "Column " << i << " added." << std::endl;
    }

    // Log and add constraints
    for (int i = 0; i < dims; ++i) {
        std::vector<HighsInt> indices;
        std::vector<double> values;
        std::cout << "Adding constraint " << i << ": ";

        for (int j = 0; j < dims; ++j) {
            if (A_ub[i * dims + j] != 0) {
                indices.push_back(j);
                values.push_back(A_ub[i * dims + j]);
                std::cout << "(" << j << ", " << A_ub[i * dims + j] << ") ";
            }
        }

        highs.addRow(-std::numeric_limits<double>::infinity(), b_ub[i], indices.size(), indices.data(), values.data());
        std::cout << " with bounds [" << -std::numeric_limits<double>::infinity() << ", " << b_ub[i] << "]" << std::endl;
    }

    std::cout << "Running solver..." << std::endl;
    HighsStatus status = highs.run();

    if (status == HighsStatus::kOk) {
        const auto& solution = highs.getSolution();
        std::copy(solution.col_value.begin(), solution.col_value.end(), result->x);
        result->fun = highs.getObjectiveValue();
        result->success = 1;
        std::cout << "Solution found with objective: " << result->fun << std::endl;
    } else {
        result->success = 0;
        std::cout << "Failed to find solution, status: " << static_cast<int>(status) << std::endl;
    }

    return result;
}
}
