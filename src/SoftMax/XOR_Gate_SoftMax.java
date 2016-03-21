package SoftMax;



import java.util.Random;
public class XOR_Gate_SoftMax {

    private static double learning_rate = 0.1;
    public static Random rand = new Random(1);

    public static void main(String[] args) throws Exception {

        double[] c_1 = {0, 0, 1};
        double[] c_3 = {1, 0, 1};
        double[] c_2 = {0, 1, 1};
        double[] c_4 = {1, 1, 1};

        double[] t_1 = {0, 1};
        double[] t_2 = {1, 0};
        double[] t_3 = {1, 0};
        double[] t_4 = {0, 1};

        double[][] data_list = {c_1, c_2, c_3, c_4};
        double[][] labl_list = {t_1, t_2, t_3, t_4};

        double[] _1_w1x= {getRand(), getRand(), getRand()};
        double[] _1_w2x= {getRand(), getRand(), getRand()};
        double[] _1_w3x= {getRand(), getRand(), getRand()};
        double[] _1_w4x= {getRand(), getRand(), getRand()};
        double[] _1_w5x= {getRand(), getRand(), getRand()};

        double[] _2_w1x= {getRand(), getRand(), getRand(), getRand(), getRand(), getRand() };
        double[] _2_w2x= {getRand(), getRand(), getRand(), getRand(), getRand(), getRand() };


        double[][] W_1 = {_1_w1x, _1_w2x, _1_w3x, _1_w4x, _1_w5x} ;

        double[][] W_2 = {_2_w1x, _2_w2x};

        Matrix data = new Matrix(data_list);
        Matrix label = new Matrix(labl_list);
        Weights weights = new Weights(W_1, W_2);



        for (int i = 0; i < 1000; i++) {
            weights = training(weights, data, label);
            print_err(weights, data, label);
        }
        print_err(weights, data, label);
        print_out(weights, data, label);
    }

    private static double getRand() {
        return rand.nextGaussian()  * .7 ;
    }

    private static void print_out(Weights weights, Matrix data, Matrix lable) throws Exception {


        Matrix w = weights.W1;
        Matrix out = data.multiplication(w.transpose());
        Matrix out_relu = out.to_ReLU();
        Matrix out_relu_append_1 = out_relu.append_1();
        Matrix w2 = weights.W2;
        Matrix out2 = out_relu_append_1.multiplication(w2.transpose());
        Matrix out_softmax = out2.to_softmax();

        double[][] output = out_softmax.A;

        for (int i = 0; i < output.length; i++) {
            System.out.print(output[i][0]);

            for (int j = 1; j < output[i].length; j++) {
                System.out.print(", "+output[i][j]);

            }

            System.out.println();
        }
        


    }


    private static double print_err(Weights weights, Matrix data, Matrix label) throws Exception {

        Matrix w = weights.W1;
        Matrix out = data.multiplication(w.transpose());
        Matrix out_relu = out.to_ReLU();
        Matrix out_relu_append_1 = out_relu.append_1();
        Matrix w2 = weights.W2;
        Matrix out2 = out_relu_append_1.multiplication(w2.transpose());
        Matrix out_softmax = out2.to_softmax();
        Matrix out_log = out_softmax.to_log();
        Matrix L_i_log_S_i = out_log.unit_prodoct(label).unit_prodoct(-1);
        Matrix dist_mat = L_i_log_S_i.aggregate_cols();

        double dist_sum = dist_mat.transpose().aggregate_cols().A[0][0];

        System.out.println(dist_sum);
        return dist_sum;

    }

    private static Weights training(Weights weights, Matrix data, Matrix label) throws Exception {

        //First Layer
        Matrix w1 = weights.W1;
        Matrix out = data.multiplication(w1.transpose());
        Matrix out_relu = out.to_ReLU();
        Matrix data_2 = out_relu.append_1();
        Matrix w2 = weights.W2;
        Matrix out2 = data_2.multiplication(w2.transpose());
        Matrix out_softmax = out2.to_softmax();
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

        
        int rows = w2.A.length;
        int cols = w2.A[0].length;


        double[][] w2_delta = new double[rows][cols];

        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {

                for (int N = 0; N < data_num; N++) {
                    w2_delta[i][j] += data_2.get(N, j) * dL_doij[N][i] * -1;
                }
            }
        }

        double[] w_jmdL_dwjm = new double[cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                w_jmdL_dwjm[rows] += w2.get(i,j) * w2_delta[i][j] ;
            }
        }


        rows = w1.A.length;
        cols = w1.A[0].length;


        double[][] w1_delta = new double[rows][cols];

        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                for (int N = 0; N < data_num; N++) {
                    w1_delta[i][j] += data.get(N, j) * (data_2.get(N,j) > 0 ? 1:0) * w_jmdL_dwjm[j];
                }
            }
        }

        Matrix w_1 =  w1.unit_plus(new Matrix(w1_delta).unit_prodoct(-1 * learning_rate));
        Matrix w_2 =  w2.unit_plus(new Matrix(w2_delta).unit_prodoct(-1 * learning_rate));


        return new Weights(w_1.A, w_2.A);

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

class Weights {

    public Matrix W1, W2;

    public Weights(double[][] w1x, double[][] w2x) {
        W1 = new Matrix(w1x);
        W2 = new Matrix(w2x);
    }
}