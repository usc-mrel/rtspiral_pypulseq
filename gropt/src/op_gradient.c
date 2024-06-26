#include "op_gradient.h"

/**
 * Initialize the opG struct
 * This is the operator that puts limits on gradient amplitude
 */
void cvxop_gradient_init(cvxop_gradient *opG, int N, int Naxis, double dt, double gmax, int ind_inv, int verbose) {
    opG->active = 1;
    opG->N = N;
    opG->Naxis = Naxis;
    opG->Ntotal = N * Naxis;
    opG->dt = dt;
    opG->gmax = gmax;
    opG->ind_inv = ind_inv;
    opG->verbose = verbose;
    

    cvxmat_alloc(&opG->Gfix, opG->Ntotal, 1);

    for (int i = 0; i < opG->Ntotal; i++) {
        opG->Gfix.vals[i] = -999999.0; // Flag for no limit on this point
    }

    for (int iA = 0; iA < Naxis; iA++) {
        opG->Gfix.vals[0 + iA*N] = 0.0;
        opG->Gfix.vals[N-1 + iA*N] = 0.0;
    }
}

/**
 * Sets a range of inputs to be equal to a given value, start and end are inclusive indices
 */
void cvxop_gradient_setFixRange(cvxop_gradient *opG, int start, int end, double val)
{
    if (start < 0) {start = 0;}
    if (end > (opG->Gfix.N-1)) {end = (opG->Gfix.N-1);}

    for (int i = start; i <= end; i++) {
        opG->Gfix.vals[i] = val;
    }
}


void cvxop_gradient_setGfix(cvxop_gradient *opG, int N_gfix, double *gfix) 
{
    if (N_gfix == 2) {
        opG->Gfix.vals[0] = gfix[0];
        opG->Gfix.vals[opG->Gfix.N-1] = gfix[1];
    } else if (N_gfix > 2) {
        if (N_gfix == opG->Gfix.N) {
            for (int i = 0; i < N_gfix; i++) {
                opG->Gfix.vals[i] = gfix[i]; 
            }
        } else {
            printf("ERROR: N_gfix did not equal the size of G\n");
        }
    }
}


/**
 * Takes an input vector and enforces gradient limits on it.
 * Points that are fixed get set to their fixed value, and every point over gmax gets set to gmax
 */
void cvxop_gradient_limiter(cvxop_gradient *opG, cvx_mat *xbar)
{
    for (int i = 0; i < xbar->N; i++) {
        if (opG->Gfix.vals[i] > -9999.0) {
            xbar->vals[i] = opG->Gfix.vals[i];
        }
    }

    for (int i = 0; i < xbar->N; i++) {
        if (xbar->vals[i] > 0.99*opG->gmax) {
            xbar->vals[i] = 0.99*opG->gmax;
        } else if (xbar->vals[i] < -0.99*opG->gmax) {
            xbar->vals[i] = -0.99*opG->gmax;
        }
    }

}

/**
 * This sets the default G initializer, it is a waveform that flips at the inversion time,
 * with gradient strength equals to 1% of gmax.
 * This was determined to help convergence speed slightly, there is probably something way better
 */
void cvxop_init_G(cvxop_gradient *opG, cvx_mat *G)
{
    for (int i = 0; i < G->N; i++) {
        if (i < opG->ind_inv) {
            G->vals[i] = 0.1 * opG->gmax;
        } else {
            G->vals[i] = -0.1 * opG->gmax;
        }
    }

}

/*
* Checks if any value of G is greater than Gmax
*/
int cvxop_gradient_check(cvxop_gradient *opG, cvx_mat *G)
{

    int grad_too_high = 0;
    int grad_bad = 0;

    for (int i = 0; i < G->N; i++) {
        if (fabs(G->vals[i]) > opG->gmax) {
            grad_too_high++;
            grad_bad = 1;
        }
    }

    
    if (opG->verbose>0) {  
        printf("    Gradient check: (%d)  %d\n", grad_bad, grad_too_high);
    }

    return grad_bad;
}

/*
* Returns the b-value of an input gradient waveform
*/
double cvxop_gradient_getbval(cvxop_gradient *opG, cvx_mat *G)
{
    double Gt = 0;
    double bval = 0;
    double mod = 71576597699.4529; // (GAMMA*2*pi)^2


    for (int i = 0; i < opG->N; i++) {
        if (i < opG->ind_inv) {
            Gt += G->vals[i];
        } else {
            Gt -= G->vals[i];
        }
        bval += Gt*Gt;
    }    
    bval *= (mod * opG->dt * opG->dt * opG->dt);

    return bval;
}   

int cvxop_gradient_overflowcheck(cvxop_gradient *opG, cvx_mat *G)
{
    double max_gval = 0.0;
    for (int i = 0; i < opG->Ntotal; i++) {
        double val = G->vals[i];
        if (fabs(val) > max_gval) {
            max_gval = fabs(val);
        }
    }

    if (max_gval > (2.0*opG->gmax)) {
        return 1;
    } else {
        return 0;
    }
}

/*
* Free memory
*/
void cvxop_gradient_destroy(cvxop_gradient *opG)
{
    free(opG->Gfix.vals);
}