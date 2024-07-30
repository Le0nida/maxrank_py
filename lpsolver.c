// lpsolver.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glpk.h>

void solve_lp(double *coefficients, int *hamstrings, double *bounds, double *result, int num_halfspaces, int dims) {
    // Create problem object
    glp_prob *lp;
    lp = glp_create_prob();
    glp_set_obj_dir(lp, GLP_MAX); // We are maximizing

    // Set number of constraints and variables
    glp_add_rows(lp, num_halfspaces + 1); // +1 for the normalization constraint
    glp_add_cols(lp, dims + 1);           // +1 for the slack variable

    // Add variables and bounds
    for (int i = 1; i <= dims; i++) {
        glp_set_col_bnds(lp, i, GLP_DB, bounds[2 * (i - 1)], bounds[2 * (i - 1) + 1]); // [low, high]
        glp_set_obj_coef(lp, i, 0.0); // All x_i have zero objective coefficient
    }
    // Slack variable x_d+1
    glp_set_col_bnds(lp, dims + 1, GLP_LO, 0.0, 0.0); // [0, +inf]
    glp_set_obj_coef(lp, dims + 1, -1.0); // We maximize x_d+1

    // Add constraints
    int ia[1 + (num_halfspaces + 1) * (dims + 1)];
    int ja[1 + (num_halfspaces + 1) * (dims + 1)];
    double ar[1 + (num_halfspaces + 1) * (dims + 1)];

    int counter = 1;
    for (int j = 1; j <= num_halfspaces; j++) {
        if (hamstrings[j - 1] == 0) {
            for (int i = 1; i <= dims; i++) {
                ia[counter] = j; ja[counter] = i; ar[counter] = -coefficients[(j - 1) * dims + (i - 1)];
                counter++;
            }
            glp_set_row_bnds(lp, j, GLP_UP, 0.0, -coefficients[(j - 1) * dims + dims]); // <= d_j
        } else {
            for (int i = 1; i <= dims; i++) {
                ia[counter] = j; ja[counter] = i; ar[counter] = coefficients[(j - 1) * dims + (i - 1)];
                counter++;
            }
            glp_set_row_bnds(lp, j, GLP_UP, 0.0, coefficients[(j - 1) * dims + dims]); // <= d_j
        }
    }

    // Normalization constraint
    for (int i = 1; i <= dims; i++) {
        ia[counter] = num_halfspaces + 1; ja[counter] = i; ar[counter] = 1.0;
        counter++;
    }
    glp_set_row_bnds(lp, num_halfspaces + 1, GLP_UP, 0.0, 1.0); // sum(x_i) <= 1

    glp_load_matrix(lp, counter - 1, ia, ja, ar);

    // Solve the problem
    glp_simplex(lp, NULL);

    // Retrieve results
    if (glp_get_status(lp) == GLP_OPT) {
        for (int i = 0; i < dims; i++) {
            result[i] = glp_get_col_prim(lp, i + 1);
        }
        result[dims] = glp_get_col_prim(lp, dims + 1); // Slack variable
    } else {
        // In case of infeasibility, return NaN or a specific code
        for (int i = 0; i <= dims; i++) {
            result[i] = -1.0;
        }
    }

    // Free memory
    glp_delete_prob(lp);
}
