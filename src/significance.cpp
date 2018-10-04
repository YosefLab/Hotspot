
#include <Rcpp.h>
using namespace Rcpp;

/*
 * Probability of k successes in n trials with success probability p
 */
// [[Rcpp::export]]
double binom_pmf(int k, int n, double p) {

    if (k > n)
        return 0.0;

    if (k < 0)
        return 0.0;

    double nCk = 1.0;

    for (int i = 1; i <= k; i++){
        nCk = nCk * (double(n+1-i) / i);
    }

    double result = nCk * pow(p,k) * pow(1-p, n-k);

    return result;
}

// [[Rcpp::export]]
double norm_sf(double x, double mu, double std){

    double zx = (x-mu) / (std * sqrt(2));

    double sf = 0.5 * erfc(zx);

    return sf;

}

/*
    What are the expected X and sd(X) where X is the sum
    of N items from a normal with mean=mu, sd=sd
*/
// [[Rcpp::export]]
double dist_params_mu(double mu, double sd, int N){

    return mu*N;

}

// [[Rcpp::export]]
double dist_params_sd(double mu, double sd, int N){

    if(N == 0)
        return 0.0;

    return sqrt(N)*sd;

}

// [[Rcpp::export]]
double nonzero_mean(NumericVector arr){
    int N = arr.size();
    double nz_total = 0;
    int nz_count = 0;

    for(int i = 0; i < N; i++){
        if(arr[i] != 0){
            nz_total += arr[i];
            nz_count += 1;
        }
    }
            
    if(nz_count == 0)
        return 0.0;
            
    return nz_total / nz_count;
}

// [[Rcpp::export]]
double nonzero_std(NumericVector arr, double mu){
    int N = arr.size();
    double var_total = 0;
    int var_count = 0;

    for(int i = 0; i < N; i++){
        if(arr[i] != 0){
            var_total += pow(arr[i] - mu,2);
            var_count += 1;
        }
    }
            
    if(var_count == 0)
        return 0.0;
    
    return sqrt(var_total / var_count);
}


// [[Rcpp::export]]
NumericVector gi_significance(
        NumericVector gi,
        NumericVector logexp,
        NumericVector counts,
        NumericMatrix neighbors,
        NumericVector umis
        ){
    
    int N_CELLS = gi.size();
    int N_NEIGHBORS = neighbors.ncol();
    
    double mu = nonzero_mean(logexp);
    double std = nonzero_std(logexp, mu);
    
    double proportion = 0.0;
    
    for(int i = 0; i < N_CELLS; i++){
        proportion += counts[i] / umis[i];
    }
        
    proportion = proportion / N_CELLS; // get average proportion
    
    // precompute the dist_params
    NumericVector e_mu(N_NEIGHBORS+1);
    NumericVector e_sd(N_NEIGHBORS+1);

    for(int k = 0; k < N_NEIGHBORS+1; k++){
        e_mu[k] = dist_params_mu(mu, std, k);
        e_sd[k] = dist_params_sd(mu, std, k);
    }

    NumericVector pvals(N_CELLS);
    
    for(int i = 0; i < N_CELLS; i++){
        
        // Get neighborhood average umi
        double ave_umi = 0;
        for(int j = 0; j < N_NEIGHBORS; j++) {
            ave_umi += umis[neighbors(i, j)-1];
        }
            
        ave_umi /= N_NEIGHBORS;
        
        // Get estimated neighborhood detect proportion
        double detect_p = 1 - binom_pmf(0, ave_umi, proportion);
        
        // Compute pval for cell
        double pval = 0.0;
        
        for(int k = 0; k < N_NEIGHBORS+1; k++){
            double pk = binom_pmf(k, N_NEIGHBORS, detect_p);
            
            if(k == 0) {
                if(gi[i] == 0)
                    pval += pk;
            } else {
                pval += norm_sf(gi[i], e_mu[k], e_sd[k]) * pk;
            }
        }
                
        pvals[i] = pval;
    }
        
    return pvals;
}

// [[Rcpp::export]]
NumericMatrix gi_significance_all(
        NumericMatrix gi,
        NumericMatrix logexp,
        NumericMatrix counts,
        NumericMatrix neighbors,
        NumericVector umis
        ){

    int N_GENES = gi.nrow();
    int N_CELLS = gi.ncol();
    NumericMatrix pvals(N_GENES, N_CELLS);

    NumericVector pvals_row;
    NumericVector gene_gi;
    NumericVector gene_logexp;
    NumericVector gene_counts;

    for(int i = 0; i < N_GENES; i++){
        gene_gi = gi(i, _);
        gene_logexp = logexp(i, _);
        gene_counts = counts(i, _);
        pvals_row = gi_significance(gene_gi, gene_logexp,
                        gene_counts, neighbors, umis);

        for(int j = 0; j < N_CELLS; j++){
            pvals(i, j) = pvals_row[j];
        }

    }

    return pvals;

}
