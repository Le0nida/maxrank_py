#include "../HiGHS/src/Highs.h"
#include <vector>
#include <cstring>  // Aggiungi questa libreria per strncpy

extern "C" {
    struct LinprogResult {
        double* x;
        double fun;
        int status;
        char message[128];
    };

    LinprogResult* linprog_highs(const double* c, const double* A_ub, const double* b_ub,
                                 const double* bounds, int num_vars, int num_constraints) {
        LinprogResult* result = new LinprogResult;

        Highs highs;

        // Define the problem dimensions
        const int num_col = num_vars;
        const int num_row = num_constraints;
        const int num_nz = num_constraints * num_vars;

        // Objective function coefficients
        std::vector<double> col_cost(c, c + num_col);

        // Constraint coefficients
        std::vector<int> A_start(num_col);
        std::vector<int> A_index(num_nz);
        std::vector<double> A_value(A_ub, A_ub + num_nz);

        for (int i = 0; i < num_col; ++i) {
            A_start[i] = i * num_row;
        }

        // Right-hand side values (upper bounds)
        std::vector<double> row_upper(b_ub, b_ub + num_row);
        // Left-hand side values (lower bounds)
        std::vector<double> row_lower(num_row, -kHighsInf);

        // Variable bounds
        std::vector<double> col_lower(num_col);
        std::vector<double> col_upper(num_col);

        for (int i = 0; i < num_col; ++i) {
            col_lower[i] = bounds[2 * i];
            col_upper[i] = bounds[2 * i + 1];
        }

        // Add columns and rows to HiGHS
        highs.addCols(num_col, col_cost.data(), col_lower.data(), col_upper.data(),
                      0, nullptr, nullptr, nullptr);
        highs.addRows(num_row, row_lower.data(), row_upper.data(),
                      num_nz, A_start.data(), A_index.data(), A_value.data());

        // Run HiGHS
        highs.run();

        // Get solution
        HighsSolution solution = highs.getSolution();
        HighsModelStatus model_status = highs.getModelStatus();

        // Prepare the result
        result->x = new double[num_col];
        for (int i = 0; i < num_col; ++i) {
            result->x[i] = solution.col_value[i];
        }
        result->fun = highs.getObjectiveValue();
        result->status = static_cast<int>(model_status);
        strncpy(result->message, highs.modelStatusToString(model_status).c_str(), 128);

        return result;
    }

    void free_linprog_result(LinprogResult* result) {
        delete[] result->x;
        delete result;
    }
}
