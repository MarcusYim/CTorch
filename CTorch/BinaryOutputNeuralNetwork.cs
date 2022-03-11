using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace CTorch
{
    class BinaryOutputNeuralNetwork
    {
        Matrix<double>[] weights;
        Vector<double>[] biases;

        int[] nodesPerLayer;
        int numLayers;

        public BinaryOutputNeuralNetwork(int[] layerNumbers)
        {
            nodesPerLayer = layerNumbers;
            numLayers = nodesPerLayer.GetLength(0);

            var mb = DenseMatrix.Build;
            var vb = DenseVector.Build;
            weights = new Matrix<double>[numLayers];
            biases = new Vector<double>[numLayers];

            weights[0] = mb.Random(layerNumbers[0], layerNumbers[0]);
            biases[0] = vb.Random(layerNumbers[0]);

            for (int i = 1; i < numLayers; i++)
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

        public double sigmoid(double x)
        {
            return 1 / (1 + Math.Pow(Constants.E, -1 * x));
        }

        public Vector<double>[,] feedForward(Vector<double> input)
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
        
        private Vector<double> outputError(Vector<double> solution)
        {

        }

        public Vector<double>[] backPropagation()
        {

        }

        public static void Main(String[] args)
        {
            long then = DateTimeOffset.Now.ToUnixTimeMilliseconds();

            int[] inp = { 10, 10, 1 };

            Classifier nn = new Classifier(inp);

            
            double[] z = { 2, 1, 5, 5, 3, 10, 34, 1, 5, 10 };
            Vector<double>[,] x = nn.feedForward(DenseVector.Build.DenseOfArray(z));
            Console.WriteLine(DateTimeOffset.Now.ToUnixTimeMilliseconds() - then + " ms");

            Console.WriteLine(x[0, x.GetLength(1) - 1]);
            Console.WriteLine(x[1, x.GetLength(1) - 1]);


            double[] arr = x[1, x.GetLength(1) - 1].ToArray();
            Console.WriteLine(nn.sigmoid(arr[0]));
        }
    }
}


