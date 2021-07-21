#include <stdlib.h>
#include <string.h>
#include "setBias.h"

int doBiasMgmt(runState *s) {
    if(s->time_step%100 == 0){
        return 1;
    }
    return 0;
}

void biasMgmt(runState *s) {
    printf("Executing management SNIP to change Bias\n");
    // Get channels
    int biasChannelID = getChannelID("biasChannel");
    int tChannelID  = getChannelID("timeChannel");
    if( biasChannelID == -1 || tChannelID == -1) {
        printf("Invalid Channel ID\n");
        return;
    }

    // get the value
    int bias;
    readChannel(biasChannelID,&bias,1);
    // set the value
    // "nxCompartment" is a reserved key word. nxCompartment[0] refers to first compartment created in Python at NxNet level.

    nxCompartment[0].Bias = bias;
/*
    int i;
    for (i=0;i<100;++i){
        nxCompartment[i].Bias = bias;
        nxCompartment[i].BiasExp = 6;
    }
*/

    // Read time of modification and send back to host
    int time = s->time_step;
    writeChannel(tChannelID, &time, 1);

    printf("Set bias=%d at t=%d\n", bias, time);
}
