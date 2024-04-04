#include "simple.h"
#include "verify.h"


using namespace Neuropia;
using namespace NeuropiaSimple;

bool SimpleTrainer::train() {
    bool ok = m_trainer.next();
    if(!ok)
        m_onEnd(m_trainer.network(), m_trainer.isOk());
    return ok;
    }



bool SimpleVerifier::verify() {
    bool ok = m_verifier.next();
    return ok;
}


