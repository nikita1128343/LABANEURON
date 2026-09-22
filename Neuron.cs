using static System.Math;

namespace PI32_Tymkul_KASHALOT.NeuroNet
{
    class Neuron
    {
        //поля
        private NeuronType type;
        private double[] weights;
        private double[] inputs;
        private double output;
        private double derivative;

        //константы для функции активации
        private double a;

        //свойства

        public double[] Weights { get => weights; set => weights = value; }

        public double[] Inputs { get => inputs; set => inputs = value; }

        public double Output { get => output; }

        public double Derivative { get => derivative; }

        //Конструктор
        public Neuron(double[] memoryWeights, NeuronType typeNeuron)
        {
            type = typeNeuron;
            weights = memoryWeights;
            inputs = new double[memoryWeights.Length];
            a = 1.0; // коэффициент крутизны сигмоиды
            output = 0.0;
            derivative = 0.0;
        }

        /// <summary>
        /// Логистическая (сигмоидальная) функция активации
        /// f(x) = 1 / (1 + e^(-a*x))
        /// </summary>
        private double ActivationFunction(double x)
        {
            return 1.0 / (1.0 + Exp(-a * x));
        }

        /// <summary>
        /// Производная логистической функции активации
        /// f'(x) = a * f(x) * (1 - f(x))
        /// Вычисляется через уже посчитанный output
        /// </summary>
        private double ActivationDerivative(double outputValue)
        {
            return a * outputValue * (1.0 - outputValue);
        }

        /// <summary>
        /// Вычисляет взвешенную сумму входов (net input)
        /// net = Σ(wi * xi)
        /// </summary>
        private double CalculateNetInput()
        {
            double net = 0.0;
            for (int i = 0; i < weights.Length && i < inputs.Length; i++)
            {
                net += weights[i] * inputs[i];
            }
            return net;
        }

        /// <summary>
        /// Вычисляет выход нейрона: применяет функцию активации к взвешенной сумме
        /// </summary>
        public double CalculateOutput()
        {
            double net = CalculateNetInput();
            output = ActivationFunction(net);
            return output;
        }

        /// <summary>
        /// Вычисляет производную функции активации для обратного распространения ошибки
        /// </summary>
        public double CalculateDerivative()
        {
            derivative = ActivationDerivative(output);
            return derivative;
        }

        /// <summary>
        /// Полный расчёт нейрона: выход + производная
        /// </summary>
        public void Process()
        {
            CalculateOutput();
            CalculateDerivative();
        }
    }
}
