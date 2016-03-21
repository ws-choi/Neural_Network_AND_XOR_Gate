package SoftMax;

import SoftMax.Matrix;

import java.util.List;


public class OR_Gate_SoftMax {
    private static double learning_rate = 0.1;

    public static void main(String[] args) throws Exception {

        double[] c_1 = {0, 0, 1};
        double[] c_2 = {0, 1, 1};
        double[] c_3 = {1, 0, 1};
        double[] c_4 = {1, 1, 1};

        double[] t_1 = {0, 1};
        double[] t_2 = {1, 0};
        double[] t_3 = {1, 0};
        double[] t_4 = {1, 0};

        double[][] data_list = {c_1, c_2, c_3, c_4};
        double[][] labl_list = {t_1, t_2, t_3, t_4};

        double[] w1x= {-0.54, 0.61 , -0.15};
        double[] w2x= {0.24, -0.24, 0.14};
        double[][] W = {w1x, w2x};

        Matrix data = new Matrix(data_list);
        Matrix label = new Matrix(labl_list);
        Matrix weights = new Matrix(W);



        for (int i = 0; i < 20000; i++) {
            weights = training(weights, data, label);
            print_err(weights, data, label);
        }
        print_err(weights, data, label);
        print_out(weights, data, label);
    }

    private static void print_out(Matrix w, Matrix data, Matrix lable) throws Exception {

        Matrix out = data.multiplication(w.transpose());
        Matrix out_softmax = out.to_softmax();

        double[][] output = out_softmax.A;

        for (int i = 0; i < output.length; i++) {
            System.out.print(output[i][0]);

            for (int j = 1; j < output[i].length; j++) {
                System.out.print(", "+output[i][j]);

            }

            System.out.println();
        }
        


    }

    private static double[][] transpose(double[][] doubles) {
        return new double[0][];
    }

    private static void print_out(double[][] w, double[][] list) {


    }





    private static double print_err(Matrix w, Matrix data, Matrix label) throws Exception {

        Matrix out = data.multiplication(w.transpose());

        Matrix out_softmax = out.to_softmax();

        Matrix out_log = out_softmax.to_log();

        Matrix L_i_log_S_i = out_log.unit_prodoct(label).unit_prodoct(-1);

        Matrix dist_mat = L_i_log_S_i.aggregate_cols();

        double dist_sum = dist_mat.transpose().aggregate_cols().A[0][0];

        System.out.println(dist_sum);
        return dist_sum;

    }

    private static Matrix training(Matrix w, Matrix data, Matrix label) throws Exception {


        Matrix out = data.multiplication(w.transpose());
        Matrix out_softmax = out.to_softmax();
        Matrix out_log = out_softmax.to_log();
        Matrix L_i_log_S_i = out_log.unit_prodoct(label).unit_prodoct(-1);
        Matrix dist_mat = L_i_log_S_i.aggregate_cols();

        int data_num = label.A.length;
        int dimension_size = label.A[0].length;


        double[][] dL_doij = new double[data_num][dimension_size];

        for (int i = 0; i < data_num; i++) {
            for (int j = 0; j < dimension_size; j++) {
                dL_doij[i][j] = (1-out_softmax.get(i,j)) * label.get(i,j) ;
            }
        }

        
        int rows = w.A.length;
        int cols = w.A[0].length;


        double[][] delta = new double[rows][cols];

        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {

                for (int N = 0; N < data_num; N++) {
                    delta[i][j] += data.get(N, j) * dL_doij[N][i] * -1;
                }
            }
        }
        

        //Matrix new_w = w.unit_plus(new Matrix(delta).unit_prodoct(-1 * learning_rate));

        return w.unit_plus(new Matrix(delta).unit_prodoct(-1 * learning_rate));
    }

    private static double sigm(double z) {
        return (1/(1+ Math.exp(-1 * z)));
    }

    private static void print_all(double[] new_w) {
        for (int i = 0; i < new_w.length; i++) {
            System.out.print(new_w[i]+", ");
        }
        System.out.println();
    }

    private static double product(double[] cases, double[] w) {

        double res = 0;
        for (int i = 0; i < cases.length; i++) {
            res += cases[i] * w[i];
        }

        return res;
    }
}