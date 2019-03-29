using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace test
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }
        double w1, w2, b;

       

        double newWeight1(double x1, double w1, double x2, double w2, double b)
        {
            // compute output
            double y;
            double a = w1 * x1 + w2 * x2 + b;

            double temp = 0.3 * x1 + 2;
            if (temp < x2)
                y = 1;
            else
                y = 0;
            double gradw = -1 * (y - a) * x1;
            w1 = w1 - 0.01 * gradw;
            return w1;
        }
        double newWeight2(double x1, double w1, double x2, double w2, double b)
        {
            // compute output
            double y;
            double a = w1 * x1 + w2 * x2 + b;

            double temp = 0.3 * x1 + 2;
            if (temp < x2)
                y = 1;
            else
                y = 0;
            double gradw = -1 * (y - a) * x2;
            w2 = w2 - 0.01 * gradw;
            return w2;
        }

        double newBias(double x1, double w1, double x2, double w2, double b)
        {
            // compute output
            double y;
            double a = w1 * x1 + w2 * x2 + b;

            double temp = 0.3 * x1 + 2;
            if (temp < x2)
                y = 1;
            else
                y = 0;
            double gradb = -1 * (y - a) * 1;
            b = b - 0.01 * gradb;
            return b;
        }

        private void button2_Click(object sender, EventArgs e)
        {
            string result = "";
            double[,] exampleData = { { 1.5, 2.4 }, { 1.5, 2.5 }, { 2.5, 2.7 }, { 2.5, 2.9 } };//4,2
            for (int i = 0; i < 4; i++)
            {
                double prob = computeAnswer(exampleData[i, 0], exampleData[i, 1]);
                result += ("X1=" + exampleData[i, 0] + " X2=" + exampleData[i, 1] + " Expected output=" + prob + "\n");
            }
            MessageBox.Show(result);
        }

        private void button1_Click(object sender, EventArgs e)
        {
            w1 = 0.3;
            w2 = 0.1;
            b = -0.1;

            double[,] trainingData = { { 1.0, 2.2 }, { 1.0, 2.4 }, { 2.0, 2.5 }, { 2.0, 2.7 }, { 3, 2.8 }, { 3, 3 }, { 4, 3.1 }, { 4, 3.3 }, { 5, 3.4 }, { 5, 3.6 }, { 6, 3.7 }, { 6, 3.8 } };//4,2

            for (int j = 0; j < 1000; j++)
            {
                for (int i = 0; i < 12; i++)
                {
                    double newW1 = newWeight1(trainingData[i, 0], w1, trainingData[i, 1], w2, b);
                    double newW2 = newWeight2(trainingData[i, 0], w1, trainingData[i, 1], w2, b);

                    double newb = newBias(trainingData[i, 0], w1, trainingData[i, 1], w2, b);
                    w1 = newW1;
                    w2 = newW2;

                    b = newb;
                }
            }
            MessageBox.Show("w1=" + w1.ToString() + " w2=" + w2.ToString() + " b=" + b.ToString());
        }

        
        private double computeAnswer(double x1, double x2)
        {
            double a = w1 * x1 + w2 * x2 + b;
            return a;
        }
    }
}
