using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace CTorch
{
    class NeuralNetwork
    {
        Matrix<double>[] weights;
        Vector<double>[] biases;

        int[] nodesPerLayer;
        int numLayers;

        public NeuralNetwork(int[] layerNumbers, int depth)
        {
            nodesPerLayer = layerNumbers;
            numLayers = depth;

            var mb = DenseMatrix.Build;
            var vb = DenseVector.Build;
            weights = new Matrix<double>[depth];
            biases = new Vector<double>[depth];

            for (int i = 0; i < depth; i++)
            {
                weights[i] = mb.Random(layerNumbers[i], layerNumbers[i - 1]);
                biases[i] = vb.Random(layerNumbers[i]);
            }
        }

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

        //this is d(relu)/dy
        //thanks william
        private double dreludy(double x)
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

        public Vector<double>[,] calculateActivationsAndZ(Vector<double> input)
        {
            Vector<double>[,] activationsAndZ = new Vector<double>[2, numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                activationsAndZ[0, i] = DenseVector.Build.Dense(nodesPerLayer[i]);
                activationsAndZ[1, i] = DenseVector.Build.Dense(nodesPerLayer[i]);
            }

            activationsAndZ[1, 0] = weights[0].Multiply(input).Add(biases[0]);
            activationsAndZ[0, 0] = activationsAndZ[1, 0].Map(x => relu(x), Zeros.AllowSkip);

            for (int i = 1; i < numLayers; i++)
            {
                activationsAndZ[1, i] = weights[i].Multiply(activationsAndZ[1, i - 1]).Add(biases[i]);
                activationsAndZ[0, i] = activationsAndZ[1, i].Map(x => relu(x), Zeros.AllowSkip);
            }

            return activationsAndZ;
        }

        public static void Main(String[] args)
        {
            int[] inp = { 3, 2, 2 };

            NeuralNetwork nn = new NeuralNetwork(inp, 6);

            double[] z = { 2, 1, 5 };
            Vector<double>[,] x = nn.calculateActivationsAndZ(DenseVector.Build.DenseOfArray(z));
            Console.WriteLine(x[0, 0]);
            Console.WriteLine(x[1, 0]);
        }
    }
}


