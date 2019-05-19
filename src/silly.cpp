#include <iostream>
#include "matrix.h"

using namespace Neuropia;

const auto sigmoidFunction = [](double value) -> double{
    return 1.0 / (1.0 + std::exp(-value));
};


template <typename ...Arg>
void print(const std::vector<double>& values){
    for(auto v : values)
        std::cout << v << " ";
    std::cout << std::endl;
}


template <typename T, typename ...Arg>
void print(const T& v){
   std::cout << v;
   std::cout << std::endl;
}

//needs pre declaration
template <typename T, typename ...Arg>
void print(const T& v, Arg... args);

template <typename ...Arg>
void print(const std::vector<double>& values, Arg...args){
    for(auto v : values)
        std::cout << v << " ";
    print(args...);
}

template <typename T, typename ...Arg>
void print(const T& v, Arg... args){
   std::cout << v << " ";
   print(args...);
}






class SillyNN {
public:
    SillyNN(int input_nodes_, int hidden_nodes_, int output_nodes_) :
        input_nodes(input_nodes_),
        hidden_nodes(hidden_nodes_),
        output_nodes(output_nodes_),
        weights_ih(Matrix<double>(input_nodes, hidden_nodes)),
        weights_ho(Matrix<double>(hidden_nodes, output_nodes)),

       bias_h(Matrix<double>(1, hidden_nodes)),
       bias_o(Matrix<double>(1, output_nodes)){
        weights_ih.randomize();
        bias_h.randomize();
        weights_ho.randomize();
        bias_o.randomize();
    }
    std::vector<double> feed(const std::vector<double>& in) const;
    void train(const std::vector<double>& input_array, const std::vector<double>& target_array);
private:
    int input_nodes;
    int hidden_nodes;
    int output_nodes;
    Matrix<double> weights_ih;
    Matrix<double> weights_ho;
    Matrix<double>  bias_h;
    Matrix<double>  bias_o;
    double learning_rate = 0.05;
};


std::vector<double> SillyNN::feed(const std::vector<double>& input_array) const {
    // Generating the Hidden Outputs
    const auto inputs = Matrix<double>::fromArray<std::vector<double>>(input_array, Matrix<double>::VecDir::row);
    auto hidden = Matrix<double>::multiply(weights_ih, inputs);
    hidden +=  bias_h;
   // activation function!
    hidden = hidden.map(sigmoidFunction);

    // Generating the output's output!
    auto output = Matrix<double>::multiply(weights_ho, hidden);
    output += bias_o;
    output = output.map(sigmoidFunction);

    assert(output.isValid());
        // Sending back to the caller!
    return output.toVector();
}

void SillyNN::train(const std::vector<double>& input_array, const std::vector<double>& target_array) {

       // Generating the Hidden Outputs
       const auto inputs = Matrix<double>::fromArray(input_array, Matrix<double>::VecDir::row);
       auto hidden = Matrix<double>::multiply(weights_ih, inputs);
       hidden += bias_h;
       // activation function!
       hidden = hidden.map(sigmoidFunction);

       // Generating the output's output!
       auto outputs = Matrix<double>::multiply(weights_ho, hidden);
       outputs += bias_o;
       outputs = outputs.map(sigmoidFunction);

       // Convert array to matrix object
       const auto targets = Matrix<double>::fromArray(target_array);

       // Calculate the error
       // ERROR = TARGETS - OUTPUTS
       const auto output_errors = targets - outputs;

       // let gradient = outputs * (1 - outputs);
       // Calculate gradient
       auto gradients = outputs.map([](double y){return y * (1 - y);});
       gradients = gradients * output_errors;
       gradients = gradients * learning_rate;

       // Adjust the bias by its deltas (which is just the gradients)
       bias_o += gradients;

       // Calculate deltas
       const auto hidden_T = hidden.transpose();
       const auto weight_ho_deltas = Matrix<double>::multiply(gradients, hidden_T);

       // Adjust the weights by deltas
       weights_ho += weight_ho_deltas;

       // Calculate the hidden layer errors
       const auto who_t = weights_ho.transpose();
       const auto hidden_errors = Matrix<double>::multiply(who_t,  output_errors);

       assert(hidden_errors.isValid());

       // Calculate hidden gradient
       auto hidden_gradient = hidden.map([](double y){return y * (1 - y);});
       hidden_gradient = hidden_gradient * hidden_errors;
       hidden_gradient = hidden_gradient * learning_rate;

       // Calcuate input->hidden delta
       auto inputs_T = inputs.transpose();
       auto weight_ih_deltas = Matrix<double>::multiply(hidden_gradient, inputs_T);

       assert(weight_ih_deltas.isValid());

       weights_ih += weight_ih_deltas;
       // Adjust the bias by its deltas (which is just the gradients)
       bias_h += hidden_gradient;

       assert(weights_ih.isValid());
       assert(bias_h.isValid());

       // outputs.print();
       // targets.print();
       // error.print();
     }

void testSilly() {

     const std::vector<std::tuple<std::vector<double>, std::vector<double>>> training_data = {{
       {0, 0},
       {0}
     }, {
       {1, 0},
       {1}
     }, {
       {0, 1},
       {1}
     }, {
       {1, 1},
       {0}
     }};

    const auto seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
     std::default_random_engine gen(seed);

    SillyNN nn(2, 2, 1);
    auto data = training_data;
     for (int i = 0; i < 50000; i++) {
         const unsigned index =
  #ifndef PSRAND
                 gen()
  #else

          NTest::ipsrand()
  #endif
                 % data.size();
         nn.train(std::get<0>(data[index]), std::get<1>(data[index]));
       }


     for(const auto& s : training_data) {
        print("in:", std::get<0>(s), "out:", nn.feed(std::get<0>(s)), "exp", std::get<1>(s));
     }


}

