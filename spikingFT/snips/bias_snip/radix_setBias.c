#include <stdlib.h>
#include <string.h>
#include "setBias.h"

// need to be set
#define SIM_TIME 250
#define N_SAMPLES 256

// really constant constant
#define MAX_DECAY 4095

int doRunMgmt(runState *s) {
    /*
    if(s->time_step%SIM_TIME == 0){
        return 1;
    }
    return 0;
     */
}

void runMgmt(runState *s) {
    printf("Executing management snip to change Bias\n");
    // Get channels
    int biasChannelID = getChannelID("biasChannel");
    int timeChannelID  = getChannelID("timeChannel");
    //int voltageChannelID  = getChannelID("voltageChannel");
    if( biasChannelID == -1 || timeChannelID == -1) {
      printf("Invalid Channel ID\n");
      return;
    }

    // get the value
    int bias;
    readChannel(biasChannelID,&bias,1);
    int i;
    for (i=0;i<(N_SAMPLES*2);++i){
        nxCompartment[i].Bias = bias;
        if (bias > 0){
            nxCompartment[i].Decay_u = MAX_DECAY;
            //nxCompartment[i].Decay_v = 0;
        }
        // TODO I think the voltage of the compartments can also be set ----> alternative offset
        //voltages[i] = nxCompartment[i].compartmentVoltage;
        //printf("voltage=%d ", voltages[i]);
    }

    int time = s->time_step;
    writeChannel(timeChannelID, &time, 1);
    printf("Set bias=%d at t=%d\n", bias, time);
}
