#ifndef DEFAULT_H
#define DEFAULT_H

constexpr char topologyRe[] = R"(\d+(,\d+)*$)";
constexpr char activationFunctionRe[] =R"((sigmoid|relu|elu)(,(sigmoid|relu|elu))*$)";
constexpr char dropoutRateRe[] = R"(\d+\.?\d*(,\d+\.?\d*)*$)";

#define DEFAULT_PARAMS \
{"ImagesVerify", "", Neuropia::Params::File}, \
{"LabelsVerify", "", Neuropia::Params::File}, \
{"Images", "", Neuropia::Params::File}, \
{"Labels", "", Neuropia::Params::File}, \
{"Iterations", "1000", Neuropia::Params::Int}, \
{"Jobs", "1", Neuropia::Params::Int}, \
{"LearningRate", "0", Neuropia::Params::Real}, \
{"LearningRateMin", "0.02", Neuropia::Params::Real}, \
{"LearningRateMax", "0.02", Neuropia::Params::Real}, \
{"BatchSize", "800", Neuropia::Params::Int}, \
{"BatchVerifySize", "100", Neuropia::Params::Int}, \
{"Topology", "64,32", topologyRe}, \
{"MaxTrainTime", std::to_string(static_cast<int>(MaxTrainTime)), Neuropia::Params::Int}, \
{"File", "mnistdata.bin", Neuropia::Params::File}, \
{"Extra", "", Neuropia::Params::String}, \
{"Hard", "false", Neuropia::Params::Bool}, \
{"ActivationFunction", "sigmoid", activationFunctionRe}, \
{"InitStrategy", "auto", R"((auto|logistic|norm|relu)$)"}, \
{"DropoutRate", "0.0", dropoutRateRe}, \
{"TestFrequency", "9999999", Neuropia::Params::Int}, \
{"L2", "0.0", Neuropia::Params::Real}, \
{"Classes", "10", Neuropia::Params::Int} \

#endif // DEFAULT_H
