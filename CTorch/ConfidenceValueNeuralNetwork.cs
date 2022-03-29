using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Random;
using MathNet.Numerics.Distributions;

namespace CTorch
{
    class ConfidenceValueNeuralNetwork
    {
        Matrix<double>[] weights;
        Vector<double>[] biases;
        readonly int alpha = 10;

        int[] nodesPerLayer;
        int numLayers;

        public ConfidenceValueNeuralNetwork(int[] layerNumbers)
        {
            nodesPerLayer = layerNumbers;
            numLayers = nodesPerLayer.GetLength(0);

            var mb = DenseMatrix.Build;
            var vb = DenseVector.Build;
            weights = new Matrix<double>[numLayers];
            biases = new Vector<double>[numLayers];

            Random seed = new Random();
            //i like the sound of this one
            Xoshiro256StarStar xs = new Xoshiro256StarStar(seed.NextFullRangeInt32(), true);
            ContinuousUniform cu = new ContinuousUniform(0.1, 10.0, xs);

            //give random values to all weights and baises
            weights[0] = mb.Random(layerNumbers[0], layerNumbers[0], cu);
            biases[0] = vb.Random(layerNumbers[0], cu);
            for (int i = 1; i < numLayers; i++)
            {
                //matrix initializing
                weights[i] = mb.Random(layerNumbers[i], layerNumbers[i - 1], cu);
                biases[i] = vb.Random(layerNumbers[i], cu);
            }
        }

        //ReLU
        private double relu(double x)
        {
            if (x <= 0)
            {
                return 0;   
            }
            else
            {
                return x;
            }
        }
        
        //derivative of ReLU
        private double drelu(double x)
        {
            if (x <= 0)
            {
                return 0;
            }
            else
            {
                return 1;
            }
        }

        //sigmoid for output
        public double sigmoid(double x)
        {
            return 1 / (1 + Math.Pow(Constants.E, -1 * x));
        }

        //calculate activations and Zs for every layer
        //0 is activations, 1 is zs
        public Vector<double>[][] feedForward(Vector<double> input)
        {
            Vector<double>[][] activationsAndZ = new Vector<double>[2][];
            activationsAndZ[0] = new Vector<double>[numLayers];
            activationsAndZ[1] = new Vector<double>[numLayers];

            //initialize arrays
            for (int i = 0; i < numLayers; i++)
            {
                activationsAndZ[0][i] = DenseVector.Build.Dense(nodesPerLayer[i]);
                activationsAndZ[1][i] = DenseVector.Build.Dense(nodesPerLayer[i]);
            }

            //first case
            activationsAndZ[1][0] = weights[0].Multiply(input).Add(biases[0]);
            
            activationsAndZ[0][0] = activationsAndZ[1][0].Map(relu, Zeros.AllowSkip);
            //loop and calculate the activations and z values
            for (int i = 1; i < numLayers; i++)
            {
                activationsAndZ[1][i] = weights[i].Multiply(activationsAndZ[1][i - 1]).Add(biases[i]);
                activationsAndZ[0][i] = activationsAndZ[1][i].Map(relu, Zeros.AllowSkip);
            }

            return activationsAndZ;
        }
        
        //get the output error
        private Vector<double> getOutputError(Vector<double> solution, Vector<double> finalActivation, Vector<double> finalZ)
        {
            //(a-y) EWiseProd  drelu(finalZ)
            return finalActivation.Subtract(solution).PointwiseMultiply(finalZ.Map(drelu, Zeros.AllowSkip));
        }

        //propagate error through network
        public Vector<double>[] backPropagate(Vector<double> outputError, Vector<double>[] zs)
        {
            //dels organized with output error at the highest index and first layer at the lowest
            Vector<double>[] deltas = new Vector<double>[numLayers];
            deltas[deltas.Length - 1] = outputError;

            for (int i = numLayers - 2; i >= 0; i--)
            {
                //del(this)=(weights(last)^T * del(last)) EWiseProd drelu(z(this))
                deltas[i] = weights[i + 1].TransposeThisAndMultiply(deltas[i + 1]).PointwiseMultiply(zs[i].Map(drelu, Zeros.AllowSkip));
            }

            return deltas;
        }

        public void gradientDescent(Vector<double> input, Vector<double> solution)
        {
            //activations and z in a jagged array
            Vector<double>[][] aAndZ = feedForward(input);
            Vector<double> outputError = getOutputError(solution, aAndZ[0][numLayers - 1], aAndZ[1][numLayers - 1]);
            Vector<double>[] deltas = backPropagate(outputError, aAndZ[1]);

            //seems like deltas has 1 too many values
            for (int i = 0; i < numLayers; i++)
            {
                biases[i] = biases[i].Subtract(deltas[i].Multiply(alpha));
            }

            for (int i = 1; i < numLayers; i++)
            {
                //outer product looks like it should work here
                weights[i] = weights[i].Subtract(deltas[i].OuterProduct(aAndZ[0][i - 1]).Multiply(alpha));
            }
        }

        public static (Vector<double> x, Vector<double> y)[] generateMirroredInputs()
        {
            (Vector<double> x, Vector<double> y)[] ret = new (Vector<double> x, Vector<double> y)[50];

            Random r = new Random();
            var vb = DenseVector.Build;

            for (int i = 0; i < 25; i++)
            {
                double[] vectX = new double[6];
                double[] vectY = new double[] { 1.0 };

                double val1 = r.NextDouble();
                double val2 = r.NextDouble();
                double val3 = r.NextDouble();

                vectX[0] = val1;
                vectX[5] = val1;

                vectX[1] = val2;
                vectX[4] = val2;

                vectX[2] = val3;
                vectX[3] = val3;

                

                ret[i] = (vb.DenseOfArray(vectX), vb.DenseOfArray(vectY));
            }

            for (int i = 25; i < 50; i++)
            {
                double[] vectX = new double[6];
                double[] vectY = new double[] { 0.0 };

                for (int x = 0; x < 6; x++)
                {
                    vectX[x] = r.NextDouble();
                }

                ret[i] = (vb.DenseOfArray(vectX), vb.DenseOfArray(vectY));
            }

            return ret;
        }

        public static void Main(String[] args)
        {
            long then = DateTimeOffset.Now.ToUnixTimeMilliseconds();

            int[] layerNums = { 6, 4, 2, 1};

            ConfidenceValueNeuralNetwork nn = new ConfidenceValueNeuralNetwork(layerNums);

            (Vector<double> x, Vector<double> y)[] input = generateMirroredInputs();
            (Vector<double> x, Vector<double> y)[] test = generateMirroredInputs();

            for (int i = 0; i < 50; i++)
            {
                nn.gradientDescent(input[i].x, input[i].y);
            }


            Console.WriteLine(nn.feedForward(test[0].x)[0][layerNums.Length - 1]);
        }
    }
}


