using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace SingleNeuronBackProp
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        double w1, w2, b;

        private void btnTrain_Click(object sender, EventArgs e)
        {
            w1 = 0.3;
            w2 = 0.1;
            b = -0.1;
            double x = 0;
            double[] trainingX = { 1.0, 1.0, 2.0, 2.0, 3.0, 3.0 };
            double[] trainingY = { 2.2, 2.4, 2.5, 2.7, 2.8, 3.0 };

            for (int j = 0; j < 1000; j++) //Epocs
            {
                for (int i = 0; i < trainingX.Length; i++) //Training points
                {
                    x = i;
                    
                    double newW1 = newWeightW1(trainingX[i], trainingY[i], w1, w2, b);
                    double newW2 = newWeightW2(trainingX[i], trainingY[i], w1, w2, b);
                    double newb = newBias(trainingX[i], trainingY[i], w1, w2, b);
                    w1 = newW1;
                    w2 = newW2;
                    b = newb;
                    
                }
            }
            MessageBox.Show("W1 = " + w1.ToString() + "\nW2 = " + w2.ToString() + "\nb =" + b.ToString());
        }

        double newWeightW1(double x1, double x2, double w1, double w2, double b)
        {
            // compute output
            double y; //class where the point belongs 1 above 0 bellow
            double a = w1 * x1 + w2 * x2 + b;
            double line_y = 0.3 * x1 + 2;
            

            if (line_y < x2) //point bellow line
            {
                y = 1;
            }
            else //point above line
            {
                y = 0;
            }

            double gradw1 = -1 * (y - a) * x1;
            w1 = w1 - 0.01 * gradw1;

            return w1;
        }

        double newWeightW2(double x1, double x2, double w1, double w2, double b)
        {
            double y; //class where point belongs: 1 above the line, 0 bellow
            double line_y = 0.3 * x1 + 2;
            double a = w1 * x1 + w2 * x2 + b;
            if (line_y < x2) //point bellow line
            {
                y = 1;
            }
            else //point above line
            {
                y = 0;
            }
            double gradw2 = -1 * (y - a) * x2;
            w2 = w2 - 0.01 * gradw2;

            return w2;
        }

        double newBias(double x1, double x2, double w1, double w2, double b)
        {
            // compute output
            double y;
            double a = w1 * x1 + w2 * x2 + b;
            double line_y = 0.3 * x1 + 2;
            
            if (line_y < x2) //point bellow
            {
                y = 1;
            }
            else
            {
                y = 0;
            }

            double gradb = -1 * (y - a) * 1;
            b = b - 0.01 * gradb;

            return b;
        }

        private void btnTest_Click(object sender, EventArgs e)
        {
            string output = "";
            double[] TestX = { 1.5, 1.5, 2.5, 2.5 };
            double[] testY = { 2.36, 2.5, 2.7, 2.8 };

            for (int i=0; i < TestX.Length; i++)
            {
                double probability = w1 * TestX[i] + w2 * testY[i] + b;

                if (probability < 0.5)
                    probability = 0;
                else
                    probability = 1;

                output += "X1= " + TestX[i] + " X2= " + testY[i] + " Output= " + probability + "\n";
            }

            MessageBox.Show(output);
        }


    }
}
