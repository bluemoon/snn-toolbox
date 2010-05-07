#include "math.h"
#include "spike_prop.h"

#define DECAY 7
inline double e(double time){
  // time >= 0 to produce valid SRF otherwise 0
  double asrf=0;
  if (time > 0){
    //spike response function
    asrf = (time * pow(M_E, (1 - time * 0.142857143))) * 0.142857143;
  }
  return asrf;
}
