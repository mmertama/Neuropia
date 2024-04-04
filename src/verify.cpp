#include "verify.h"
#include "utils.h"
#include <map>
#include <utility>
#include <functional>
#include <vector>
		

using namespace Neuropia;
using ReadFunction = std::function<std::pair<std::vector<unsigned char>, unsigned>()>;
using VerifyFunction = std::function<size_t (const std::vector<Neuropia::NeuronType>& inputs)>;

class Verifier::Private {
    public:
    Private(const std::vector<Neuropia::Layer>& ensembles,
        bool hard,
        const std::string& imageFile,
        const std::string& labelFile,
        bool quiet,
        size_t from,
        size_t count,
        bool random) : Private(imageFile, labelFile, quiet, from, count, random) {

            verify_function = [&](const std::vector<Neuropia::NeuronType>& inputs) {
                std::map<unsigned, size_t> hardVotes; //hard we take one got most of outputs for each round
                std::map<size_t, NeuronType> softVotes;   //we sum up the results and take one get more over all results in round
                for(const auto& network : ensembles) {
                    const auto& outputs = network.feed(inputs.begin(), inputs.end());
                    if(hard) {
                        const auto result = static_cast<unsigned>(std::distance(outputs.begin(),
                                                            std::max_element(outputs.begin(), outputs.end())));
                        hardVotes[result]++;
                    } else {
                        for(auto it = outputs.begin(); it != outputs.end(); it++) {
                            softVotes[static_cast<unsigned>(std::distance(outputs.begin(), it))] += *it;
                        }
                    }
                }

                const auto comp = [](const auto& a, const auto& b){return a.second < b.second;};

                size_t max;
                if(hard) {
                    max = std::max_element(hardVotes.begin(), hardVotes.end(), comp)->first;
                } else {
                    max = static_cast<size_t>(std::distance(softVotes.begin(),
                                                        std::max_element(softVotes.begin(), softVotes.end(), comp)));
                }

                return max;
            };
        }
    Private(const Neuropia::Layer& network,
        const std::string& imageFile,
        const std::string& labelFile,
        bool quiet,
        size_t from,
        size_t count,
        bool random) : Private(imageFile, labelFile, quiet, from, count, random) {
            verify_function = [&] (const std::vector<Neuropia::NeuronType>& inputs) {
                const auto& outputs = network.feed(inputs.begin(), inputs.end());
                const auto max = static_cast<unsigned>(std::distance(outputs.begin(),
                                                        std::max_element(outputs.begin(), outputs.end())));
                return max;
            };
        }

        bool next() {
            if(m_current >= m_iterations)
                return false;
   
            const auto& [image, label] = read_function();
            const auto imageSize = m_testImages.size(1) * m_testImages.size(2);
            std::vector<Neuropia::NeuronType> inputs(imageSize);
            std::transform(image.begin(), image.end(), inputs.begin(), [](unsigned char c) {
                return Neuropia::normalize(static_cast<Neuropia::NeuronType>(c), 0, 255);
            });
            
            const auto max = verify_function(inputs);

            if(!m_quiet) {
                percentage(m_current, m_iterations);
            }
            
             if(max == label) {
                ++m_found;
            } 
            ++m_current;
            return true;                                     
        }

        auto found() const {
            return m_found;
        }

        auto result() const {
            return m_iterations > 0 ? static_cast<Neuropia::NeuronType>(m_found) / static_cast<Neuropia::NeuronType>(m_iterations) : 0.;
        }

        size_t size() const {
            return std::min(m_testLabels.size(), m_testImages.size());
        }   

    private:
        // delegate    
        Private(const std::string& imageFile,
            const std::string& labelFile,
            bool quiet,
            size_t from,
            size_t count,
            bool is_random) : m_testImages(imageFile), m_testLabels(labelFile), m_quiet(quiet) {
                
                if(!m_testImages.ok()) {
                    std::cerr << "Cannot open images from \"" << imageFile << "\" ";
                    m_testImages.perror(); 
                    std::cerr << std::endl;
                    return;
                }

                if(!m_testLabels.ok()) {
                    std::cerr << "Cannot open labels from \"" << labelFile << "\" "; 
                    m_testLabels.perror(); 
                    std::cerr << std::endl;
                    return;
                }

                const auto imageSize = m_testImages.size(1) * m_testImages.size(2);
                const auto size = std::min(m_testLabels.size(), m_testImages.size());

                const ReadFunction random_function = [this, size, imageSize]() {
                    const auto index = m_random.random(size);
                    return std::make_pair(m_testImages.readAt(index, imageSize), static_cast<unsigned>(m_testLabels.readAt(index)));
                };

                const ReadFunction seq_function = [this, imageSize]() {
                    return std::make_pair(m_testImages.read(imageSize), static_cast<unsigned>(m_testLabels.read()));
                };
    
                read_function = is_random ? random_function : seq_function;

                m_iterations =  is_random ? count : std::min(size + from, count);
            }

    private:
    
        Neuropia::IdxReader<unsigned char> m_testImages;
        Neuropia::IdxReader<unsigned char> m_testLabels;
        const bool m_quiet;
        size_t m_iterations = 0;
        ReadFunction read_function = nullptr;
        VerifyFunction verify_function = nullptr;
        Neuropia::Random m_random{}; 
        size_t m_current = 0;
        unsigned m_found = 0;
    };

    

Verifier::Verifier(const Neuropia::Layer& network,
                 const std::string& imageFile,
                 const std::string& labelFile,
                 bool quiet,
                 size_t from, // if random - ignored
                 size_t count,
                 bool random) : m_private(std::make_unique<Private>(network, imageFile, labelFile, quiet, from, count, random)) {}
             
Verifier::Verifier(const std::vector<Neuropia::Layer>& ensembles,
                   bool hard,
                   const std::string& imageFile,
                   const std::string& labelFile,
                   bool quiet,
                   size_t from,
                   size_t count,
                   bool random) : m_private(std::make_unique<Private>(ensembles, hard, imageFile, labelFile, quiet, from, count, random)) {}

bool Verifier::next() {
    return m_private->next();
}

size_t Verifier::found() const {
    return m_private->found();
}

VerifyResult Verifier::result() const {
    return { m_private->found(), m_private->result(), m_private->size()};
}

VerifyResult Verifier::busy(bool quiet) {
    const auto loop = [this]() {
        while(next());
    };
    if(quiet) {
        loop();
    } else {    
        Neuropia::timed(loop, "Verify"); 
    }
    return result();   
        
}          

Verifier::~Verifier() {}
