import sys
sys.path.insert(0,'..')
from scipy.weave import  ext_tools

def build_math():
    """ Builds an extension module with fibonacci calculators.
    """
    mod = ext_tools.ext_module('math_ext')
    time = 0.435 # this is effectively a type declaration
    e_code = """
    #include "math.h"
    #include "spike_prop.h"

    #define DECAY 7
    inline double e(double time){
    double asrf=0;
    if (time > 0){
      //spike response function
      asrf = (time * pow(M_E, (1 - time * 0.142857143))) * 0.142857143;
    }
    return asrf;
    }
    """
    
    
    ext_code = """
    return_val = e(time);
    """
    e_ = ext_tools.ext_function('e', ext_code, ['time'])
    e_.customize.add_support_code(e_code)
    mod.add_function(e_)
    mod.compile()
